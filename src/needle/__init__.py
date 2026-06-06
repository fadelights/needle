"""
Needle
======

This module exposes a small, user-friendly API for common operations
also used in the needle API.


Available functions
-------------------

import needle

needle.connect(...)
    Connect to external services (Elasticsearch, S3) and warm them up.
needle.index(...)
    Index a document: upload to object storage and run the indexing pipeline.
needle.query(...)
    Run a query against the business index and return a structured result.
needle.list_(...)
    List files stored for a business.
needle.delete(...)
    Delete an object from storage and its documents from the document store.

The functions here are thin wrappers around the implementations in the
submodules and are intentionally small to keep import-time work light.
"""

from __future__ import annotations

from .core import delete, index, list_, query
from .storage import healthcheck

# TODO: Registration and auth


def connect(timeout: int = 5) -> dict[str, bool]:
    """Health check for external services (Elasticsearch, S3)."""
    # TODO: Connection parameters and retries
    return healthcheck(timeout=timeout)


__all__ = ["connect", "index", "query", "list_", "delete"]
