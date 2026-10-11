"""
Hierarchical Parent-Child chunking engine.
Generates:
1. Large Parent Chunks (1000-1500 tokens) -> Stored in SQLite (Ground Truth)
2. Small Child Chunks (150-200 tokens)     -> Indexed in ChromaDB (Vector Search)
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Generator

from src.config import settings
from src.parser import ParsedBlock
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParentChunk:
    """Represents full contextual paragraph or raw table stored in SQLite."""
    parent_id: str
    doc_id: str
    page_number: int
    chunk_index: int
    content: str
    content_type: str  # 'text' or 'table'


@dataclass
class ChildChunk:
    """Represents fine-grained snippet indexed in Vector DB for precise similarity search."""
    child_id: str
    parent_id: str
    doc_id: str
    page_number: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalChunker:
    """Splits parsed document blocks into synchronized Parent and Child chunks."""

    def __init__(
        self,
        child_size: int = settings.CHUNK_CHILD_SIZE,
        child_overlap: int = settings.CHUNK_CHILD_OVERLAP,
        parent_size: int = settings.CHUNK_PARENT_SIZE,
    ):
        # Approximate: 1 token ≈ 4 characters
        self.child_char_limit = child_size * 4
        self.child_overlap_chars = child_overlap * 4
        self.parent_char_limit = parent_size * 4

    def chunk_document(
        self,
        doc_id: str,
        blocks_generator: Generator[ParsedBlock, None, None]
    ) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """
        Consumes streaming blocks from the parser and produces synchronized Parent & Child chunks.
        """
        parent_chunks: List[ParentChunk] = []
        child_chunks: List[ChildChunk] = []
        global_chunk_idx = 0

        logger.info(f"Starting hierarchical chunking for document: {doc_id}")

        for block in blocks_generator:
            # Table blocks are treated as self-contained parent chunks
            if block.block_type == "table":
                p_chunk, c_chunks = self._chunk_table_block(doc_id, block, global_chunk_idx)
                parent_chunks.append(p_chunk)
                child_chunks.extend(c_chunks)
                global_chunk_idx += 1
            else:
                # Narrative text blocks
                p_chunks, c_chunks = self._chunk_text_block(doc_id, block, global_chunk_idx)
                parent_chunks.extend(p_chunks)
                child_chunks.extend(c_chunks)
                global_chunk_idx += len(p_chunks)

        logger.info(
            f"Chunking completed for {doc_id}: Generated {len(parent_chunks)} Parent Chunks "
            f"and {len(child_chunks)} Child Chunks"
        )
        return parent_chunks, child_chunks

    def _chunk_text_block(
        self,
        doc_id: str,
        block: ParsedBlock,
        start_index: int
    ) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """Splits narrative text into parent windows, then derives overlapping children."""
        text = block.content.strip()
        if not text:
            return [], []

        parents: List[ParentChunk] = []
        children: List[ChildChunk] = []

        # Split text into paragraphs
        raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        current_parent_text = []
        current_parent_len = 0
        local_idx = 0

        for para in raw_paragraphs:
            para_len = len(para)
            if current_parent_len + para_len > self.parent_char_limit and current_parent_text:
                # Finish current parent chunk
                p_content = "\n\n".join(current_parent_text)
                parent_id = f"{doc_id}_p{block.page_number}_{start_index + local_idx}"
                parent = ParentChunk(
                    parent_id=parent_id,
                    doc_id=doc_id,
                    page_number=block.page_number,
                    chunk_index=start_index + local_idx,
                    content=p_content,
                    content_type="text"
                )
                parents.append(parent)
                children.extend(self._create_child_chunks(parent))

                current_parent_text = [para]
                current_parent_len = para_len
                local_idx += 1
            else:
                current_parent_text.append(para)
                current_parent_len += para_len

        # Flush any remaining text as the final parent chunk
        if current_parent_text:
            p_content = "\n\n".join(current_parent_text)
            parent_id = f"{doc_id}_p{block.page_number}_{start_index + local_idx}"
            parent = ParentChunk(
                parent_id=parent_id,
                doc_id=doc_id,
                page_number=block.page_number,
                chunk_index=start_index + local_idx,
                content=p_content,
                content_type="text"
            )
            parents.append(parent)
            children.extend(self._create_child_chunks(parent))

        return parents, children

    def _create_child_chunks(self, parent: ParentChunk) -> List[ChildChunk]:
        """Creates fine-grained overlapping child windows from a parent chunk."""
        text = parent.content
        children: List[ChildChunk] = []

        # Sentence-based boundary splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_child_sentences = []
        current_len = 0
        child_idx = 0

        for sentence in sentences:
            s_len = len(sentence)
            if current_len + s_len > self.child_char_limit and current_child_sentences:
                child_content = " ".join(current_child_sentences)
                child_id = f"{parent.parent_id}_c{child_idx}"
                children.append(ChildChunk(
                    child_id=child_id,
                    parent_id=parent.parent_id,
                    doc_id=parent.doc_id,
                    page_number=parent.page_number,
                    content=child_content,
                    metadata={"content_type": parent.content_type}
                ))
                child_idx += 1

                # Overlap: keep the last sentence for context continuity
                current_child_sentences = [current_child_sentences[-1], sentence]
                current_len = len(current_child_sentences[0]) + s_len
            else:
                current_child_sentences.append(sentence)
                current_len += s_len

        if current_child_sentences:
            child_content = " ".join(current_child_sentences)
            child_id = f"{parent.parent_id}_c{child_idx}"
            children.append(ChildChunk(
                child_id=child_id,
                parent_id=parent.parent_id,
                doc_id=parent.doc_id,
                page_number=parent.page_number,
                content=child_content,
                metadata={"content_type": parent.content_type}
            ))

        return children

    def _chunk_table_block(
        self,
        doc_id: str,
        block: ParsedBlock,
        chunk_idx: int
    ) -> Tuple[ParentChunk, List[ChildChunk]]:
        """Handles Markdown table: Parent is full raw Markdown table; Children are row groups."""
        parent_id = f"{doc_id}_p{block.page_number}_tbl_{chunk_idx}"
        parent = ParentChunk(
            parent_id=parent_id,
            doc_id=doc_id,
            page_number=block.page_number,
            chunk_index=chunk_idx,
            content=block.content,
            content_type="table"
        )

        children: List[ChildChunk] = []
        lines = block.content.strip().split("\n")
        
        # If table is small (under child limit), table itself is the single child
        if len(block.content) <= self.child_char_limit:
            children.append(ChildChunk(
                child_id=f"{parent_id}_c0",
                parent_id=parent_id,
                doc_id=doc_id,
                page_number=block.page_number,
                content=block.content,
                metadata={"content_type": "table"}
            ))
        else:
            # Split large table into row groups of 3-5 rows, repeating the header
            headers = lines[:2] if len(lines) >= 2 else []
            data_rows = lines[2:] if len(lines) >= 2 else lines
            
            row_group = []
            group_idx = 0
            for r in data_rows:
                row_group.append(r)
                if len(row_group) >= 4:
                    group_content = "\n".join(headers + row_group)
                    children.append(ChildChunk(
                        child_id=f"{parent_id}_c{group_idx}",
                        parent_id=parent_id,
                        doc_id=doc_id,
                        page_number=block.page_number,
                        content=group_content,
                        metadata={"content_type": "table"}
                    ))
                    group_idx += 1
                    row_group = []
                    
            if row_group:
                group_content = "\n".join(headers + row_group)
                children.append(ChildChunk(
                    child_id=f"{parent_id}_c{group_idx}",
                    parent_id=parent_id,
                    doc_id=doc_id,
                    page_number=block.page_number,
                    content=group_content,
                    metadata={"content_type": "table"}
                ))

        return parent, children
