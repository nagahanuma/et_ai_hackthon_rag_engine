"""
Centralized logging utility for the Compliance RAG Engine.
Includes:
- Millisecond timestamps
- Thread Name
- Dynamic Transaction ID (Correlation ID) via contextvars
- Filename, method name, and line numbers
"""

import sys
import uuid
import logging
from pathlib import Path
from contextvars import ContextVar

# ContextVar stores a transaction ID scoped to the current thread/async task
_transaction_id_ctx: ContextVar[str] = ContextVar("transaction_id", default="SYSTEM")


def set_transaction_id(txn_id: str = None) -> str:
    """Sets a new transaction ID for the current execution context."""
    if not txn_id:
        txn_id = f"txn_{uuid.uuid4().hex[:8]}"
    _transaction_id_ctx.set(txn_id)
    return txn_id


def get_transaction_id() -> str:
    """Retrieves the current transaction ID."""
    return _transaction_id_ctx.get()


def clear_transaction_id() -> None:
    """Resets the transaction ID to SYSTEM."""
    _transaction_id_ctx.set("SYSTEM")


class TransactionFilter(logging.Filter):
    """Injects the dynamic transaction ID into every log record."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.txn_id = get_transaction_id()
        return True


def get_logger(name: str = "compliance_rag") -> logging.Logger:
    """
    Returns a configured logger instance.
    Format:
    [YYYY-MM-DD HH:MM:SS.mmm] | LEVEL | [Thread] | [TXN: id] | filename.py:method:line - message
    """
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # Log format including Thread Name and Transaction ID
    log_format = (
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
        "[%(threadName)s] | [TXN: %(txn_id)s] | "
        "%(filename)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Attach the TransactionFilter
    txn_filter = TransactionFilter()

    # Console Handler (reconfigure to utf-8 for Windows PowerShell support)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(txn_filter)
    logger.addHandler(console_handler)

    # File Handler
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(logs_dir / "compliance_app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(txn_filter)
    logger.addHandler(file_handler)

    return logger
