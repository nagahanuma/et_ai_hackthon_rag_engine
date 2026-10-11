/**
 * Banking Regulatory Compliance RAG Engine - Frontend Controller
 * Handles:
 * 1. Document Lifecycle (Upload, Poll Jobs, List, Replace, Delete)
 * 2. Compliance Audit Chat with Citation Badges
 * 3. Live RAG Process Flow & Observability Trace Visualizer
 * 4. Raw Storage Inspector (SQLite Ground Truth vs. ChromaDB Vectors)
 */

function getApiUrl() {
    const input = document.getElementById("apiUrlInput");
    return input ? input.value.trim().replace(/\/$/, "") : "http://localhost:8000";
}

// ==========================================
// 1. TAB NAVIGATION
// ==========================================
function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach((pane) => pane.classList.remove("active"));

    const targetPane = document.getElementById(tabId);
    if (targetPane) targetPane.classList.add("active");

    const activeBtn = Array.from(document.querySelectorAll(".tab-btn")).find((btn) =>
        btn.getAttribute("onclick").includes(tabId)
    );
    if (activeBtn) activeBtn.classList.add("active");

    if (tabId === "tab-docs") loadDocuments();
    if (tabId === "tab-storage") loadStorageInspector();
}

// ==========================================
// 2. DOCUMENT LIFECYCLE (CRUD)
// ==========================================

async function loadDocuments() {
    const tbody = document.getElementById("documentsTableBody");
    try {
        const res = await fetch(`${getApiUrl()}/api/v1/documents`);
        if (!res.ok) throw new Error("Failed to fetch documents");
        const data = await res.json();

        if (!data.documents || data.documents.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 2rem;">No documents indexed yet. Upload a PDF or TXT file above.</td></tr>`;
            return;
        }

        tbody.innerHTML = data.documents
            .map((doc) => {
                const sizeKb = (doc.file_size_bytes / 1024).toFixed(1);
                const dateStr = new Date(doc.created_at).toLocaleString();
                return `
          <tr>
            <td><strong>📄 ${doc.filename}</strong></td>
            <td><span class="badge-tag">v${doc.version}</span></td>
            <td>${doc.total_pages}</td>
            <td>${sizeKb} KB</td>
            <td>${doc.total_parent_chunks || "-"}</td>
            <td style="color: var(--text-muted); font-size: 0.8rem;">${dateStr}</td>
            <td>
              <button class="btn btn-secondary" onclick="promptReplaceDocument('${doc.doc_id}', '${doc.filename}')">Replace</button>
              <button class="btn btn-danger" style="margin-left: 0.4rem;" onclick="deleteDocument('${doc.doc_id}', '${doc.filename}')">Delete</button>
            </td>
          </tr>
        `;
            })
            .join("");
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="7" style="color: var(--accent-red); text-align: center;">Error loading documents: ${err.message}. Is backend running?</td></tr>`;
    }
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const progressBox = document.getElementById("uploadProgressContainer");
    const progressBar = document.getElementById("uploadProgressBar");
    const statusLabel = document.getElementById("uploadStatusLabel");
    const percentLabel = document.getElementById("uploadPercentLabel");

    progressBox.style.display = "block";
    progressBar.style.width = "10%";
    statusLabel.textContent = `Streaming '${file.name}' to server...`;
    percentLabel.textContent = "10%";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("doc_type", "regulatory_circular");

    try {
        const res = await fetch(`${getApiUrl()}/api/v1/documents/upload`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || "Upload failed");
        }

        const data = await res.json();
        progressBar.style.width = "40%";
        statusLabel.textContent = "Parsing pages & generating embeddings in background...";
        percentLabel.textContent = "40%";

        // Poll until document appears in active registry
        let attempts = 0;
        const interval = setInterval(async () => {
            attempts++;
            progressBar.style.width = `${Math.min(40 + attempts * 15, 95)}%`;
            percentLabel.textContent = `${Math.min(40 + attempts * 15, 95)}%`;

            const check = await fetch(`${getApiUrl()}/api/v1/documents`);
            const checkData = await check.json();
            const found = checkData.documents.find((d) => d.filename === file.name);

            if (found || attempts > 15) {
                clearInterval(interval);
                progressBar.style.width = "100%";
                percentLabel.textContent = "100%";
                statusLabel.textContent = `Successfully indexed '${file.name}'!`;
                setTimeout(() => {
                    progressBox.style.display = "none";
                    loadDocuments();
                }, 1200);
            }
        }, 1500);
    } catch (err) {
        statusLabel.textContent = `Upload error: ${err.message}`;
        statusLabel.style.color = "var(--accent-red)";
    } finally {
        event.target.value = "";
    }
}

async function promptReplaceDocument(docId, filename) {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".pdf,.txt,.md";
    fileInput.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (!confirm(`Are you sure you want to replace '${filename}' with '${file.name}'? Old vectors will be purged.`)) {
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch(`${getApiUrl()}/api/v1/documents/${docId}/replace`, {
                method: "PUT",
                body: formData,
            });
            if (!res.ok) throw new Error("Replacement failed");
            alert(`Document successfully upgraded to a new version!`);
            loadDocuments();
        } catch (err) {
            alert(`Error replacing document: ${err.message}`);
        }
    };
    fileInput.click();
}

async function deleteDocument(docId, filename) {
    if (!confirm(`Delete '${filename}' permanently? All associated SQLite parent chunks and ChromaDB vectors will be purged.`)) {
        return;
    }
    try {
        const res = await fetch(`${getApiUrl()}/api/v1/documents/${docId}`, { method: "DELETE" });
        if (!res.ok) throw new Error("Delete failed");
        loadDocuments();
    } catch (err) {
        alert(`Error deleting document: ${err.message}`);
    }
}

// ==========================================
// 3. COMPLIANCE AUDIT CHAT
// ==========================================

async function sendQuery() {
    const input = document.getElementById("chatInput");
    const query = input.value.trim();
    if (!query) return;

    const chatMessages = document.getElementById("chatMessages");

    // Append user message
    const userDiv = document.createElement("div");
    userDiv.className = "message user";
    userDiv.innerHTML = `<p>${escapeHtml(query)}</p>`;
    chatMessages.appendChild(userDiv);
    input.value = "";
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Append thinking bot message
    const botDiv = document.createElement("div");
    botDiv.className = "message bot";
    botDiv.innerHTML = `<p><em>🔍 Searching regulatory indexes & reasoning...</em></p>`;
    chatMessages.appendChild(botDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const res = await fetch(`${getApiUrl()}/api/v1/compliance/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Query failed");
        }

        const data = await res.json();

        // Render Answer with Citations
        const citationsHtml =
            data.citations && data.citations.length > 0
                ? `<div class="citations-box">
            <span style="font-size: 0.75rem; color: var(--text-muted); width: 100%; margin-bottom: 0.2rem;">Verified Audit Citations:</span>
            ${data.citations
                    .map(
                        (c) =>
                            `<span class="citation-badge" title="Parent Chunk: ${c.parent_id}">📄 ${c.filename} (Page ${c.page_number}) [${c.content_type}]</span>`
                    )
                    .join("")}
          </div>`
                : "";

        botDiv.innerHTML = `
      <p style="white-space: pre-wrap;">${escapeHtml(data.answer)}</p>
      ${citationsHtml}
      <div style="margin-top: 0.5rem; font-size: 0.75rem; color: var(--text-muted); display: flex; justify-content: space-between;">
        <span>⏱️ Latency: ${data.execution_time_ms.toFixed(0)} ms</span>
        <a href="javascript:void(0)" onclick="switchTab('tab-flow')" style="color: var(--accent-cyan); text-decoration: none;">View Process Trace ➔</a>
      </div>
    `;

        // Send trace data to Tab 3 Inspector
        if (data.trace) {
            renderTrace(data.trace, data.query, data.execution_time_ms);
        }
    } catch (err) {
        botDiv.innerHTML = `<p style="color: var(--accent-red);">Error: ${escapeHtml(err.message)}</p>`;
    } finally {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

// ==========================================
// 4. OBSERVABILITY & PROCESS FLOW INSPECTOR
// ==========================================

function renderTrace(trace, query, totalTime) {
    document.getElementById("traceQueryBadge").textContent = `Query: "${query}"`;
    document.getElementById("traceTotalTime").textContent = `${totalTime.toFixed(0)} ms`;

    if (trace.timing_breakdown) {
        document.getElementById("timeDense").textContent = `${trace.timing_breakdown.dense_search_ms} ms`;
        document.getElementById("timeBM25").textContent = `${trace.timing_breakdown.bm25_search_ms} ms`;
        document.getElementById("timeLLM").textContent = `${trace.timing_breakdown.llm_generation_ms} ms`;
    }

    // Dense Matches
    const denseBox = document.getElementById("traceDenseMatches");
    denseBox.textContent = JSON.stringify(trace.dense_vector_matches || [], null, 2);

    // BM25 Matches
    const bm25Box = document.getElementById("traceBM25Matches");
    bm25Box.textContent = JSON.stringify(trace.bm25_keyword_matches || [], null, 2);

    // Parent Contexts
    const parentBox = document.getElementById("traceParentContexts");
    parentBox.textContent = JSON.stringify(trace.resolved_parent_contexts || [], null, 2);

    // Raw LLM Prompt
    const promptBox = document.getElementById("traceLLMPrompt");
    if (trace.full_prompt_sent_to_llm) {
        promptBox.textContent =
            `=== SYSTEM PROMPT ===\n${trace.full_prompt_sent_to_llm.system_prompt}\n\n` +
            `=== USER PROMPT ===\n${trace.full_prompt_sent_to_llm.user_prompt}`;
    }
}

// ==========================================
// 5. STORAGE INSPECTOR (SQLITE vs CHROMADB)
// ==========================================

// ==========================================
// 5. STORAGE INSPECTOR (SQLITE vs CHROMADB)
// ==========================================

async function loadStorageInspector() {
    const sqliteView = document.getElementById("sqliteStorageView");
    const chromaView = document.getElementById("chromaStorageView");

    sqliteView.innerHTML = `<div style="color: var(--text-muted); padding: 1rem;">Querying SQLite ground truth tables...</div>`;
    chromaView.innerHTML = `<div style="color: var(--text-muted); padding: 1rem;">Querying ChromaDB vectors and payloads...</div>`;

    try {
        const [resSqlite, resChroma] = await Promise.all([
            fetch(`${getApiUrl()}/api/v1/inspect/sqlite`),
            fetch(`${getApiUrl()}/api/v1/inspect/chroma`),
        ]);

        if (!resSqlite.ok) throw new Error(`SQLite Inspector returned HTTP ${resSqlite.status}`);
        if (!resChroma.ok) throw new Error(`ChromaDB Inspector returned HTTP ${resChroma.status}`);

        const dataSqlite = await resSqlite.json();
        const dataChroma = await resChroma.json();

        // 1. RENDER SQLITE RELATIONAL TABLE
        const parents = dataSqlite.tables?.parent_chunks || [];
        if (parents.length === 0) {
            sqliteView.innerHTML = `<div style="color: var(--text-muted); padding: 1rem;">No records found in SQLite. Ingest a document first.</div>`;
        } else {
            sqliteView.innerHTML = `
                <div style="margin-bottom: 0.6rem; font-size: 0.8rem; color: var(--text-muted); display: flex; justify-content: space-between;">
                    <span>Showing <b>${parents.length}</b> rows from table <code>parent_chunks</code></span>
                    <span>Total Docs: <b>${dataSqlite.tables?.documents?.length || 0}</b></span>
                </div>
                <div class="storage-table-container">
                    <table class="storage-table">
                        <thead>
                            <tr>
                                <th>parent_id (PK)</th>
                                <th>page</th>
                                <th>type</th>
                                <th>content (Legal Ground Truth)</th>
                                <th>chars</th>
                                <th>doc_id (FK)</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${parents.map(p => `
                                <tr>
                                    <td><code style="color: var(--accent-blue); font-weight: 600;">${escapeHtml(p.parent_id)}</code></td>
                                    <td><span class="badge-tag">p.${p.page_number}</span></td>
                                    <td><span style="font-size: 0.75rem; color: var(--text-muted);">${escapeHtml(p.content_type)}</span></td>
                                    <td class="cell-content" title="${escapeHtml(p.content_preview)}">${escapeHtml(p.content_preview)}</td>
                                    <td>${p.total_chars}</td>
                                    <td><code style="color: var(--text-muted); font-size: 0.75rem;">${escapeHtml(p.doc_id)}</code></td>
                                </tr>
                            `).join("")}
                        </tbody>
                    </table>
                </div>
            `;
        }

        // 2. RENDER CHROMADB VECTOR TABLE
        const vectors = dataChroma.sample_records || [];
        if (vectors.length === 0) {
            chromaView.innerHTML = `<div style="color: var(--text-muted); padding: 1rem;">No vectors found in ChromaDB. Ingest a document first.</div>`;
        } else {
            chromaView.innerHTML = `
                <div style="margin-bottom: 0.6rem; font-size: 0.8rem; color: var(--text-muted); display: flex; justify-content: space-between;">
                    <span>Collection: <code style="color: var(--accent-cyan);">${escapeHtml(dataChroma.collection_name)}</code></span>
                    <span>Total Vectors: <b>${dataChroma.total_vectors_in_collection}</b></span>
                </div>
                <div class="storage-table-container">
                    <table class="storage-table">
                        <thead>
                            <tr>
                                <th>id (Vector ID)</th>
                                <th>embedding (Float[384])</th>
                                <th>metadata.parent_id (Pointer)</th>
                                <th>metadata.page</th>
                                <th>document (Child Text Payload)</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${vectors.map(v => `
                                <tr>
                                    <td><code style="color: var(--accent-cyan); font-weight: 600;">${escapeHtml(v.id)}</code></td>
                                    <td><span class="vector-badge">${escapeHtml(v.embedding_preview)}</span></td>
                                    <td><code style="color: var(--accent-blue);">${escapeHtml(v.metadata?.parent_id || "-")}</code></td>
                                    <td><span class="badge-tag">p.${v.metadata?.page_number || "?"}</span></td>
                                    <td class="cell-content" title="${escapeHtml(v.document_snippet)}">${escapeHtml(v.document_snippet)}</td>
                                </tr>
                            `).join("")}
                        </tbody>
                    </table>
                </div>
            `;
        }



    } catch (err) {
        sqliteView.innerHTML = `<div style="color: var(--accent-red); padding: 1rem;">❌ Error: ${escapeHtml(err.message)}</div>`;
        chromaView.innerHTML = `<div style="color: var(--accent-red); padding: 1rem;">❌ Error: ${escapeHtml(err.message)}</div>`;
    }
}


function escapeHtml(text) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

// Startup
document.addEventListener("DOMContentLoaded", () => {
    loadDocuments();
});
