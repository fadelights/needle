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
needle.list(...)
    List files stored for a business.
needle.delete(...)
    Delete an object from storage and its documents from the document store.

The functions here are thin wrappers around the implementations in the
submodules and are intentionally small to keep import-time work light.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional

from haystack.dataclasses import ByteStream

from .pipelines import get_indexing_pipeline, get_query_pipeline
from .storage import get_document_store, get_s3_storage, healthcheck

# TODO: Registration and auth


def connect(timeout: int = 5) -> dict[str, bool]:
    """Health check for external services (Elasticsearch, S3)."""
    # TODO: Connection parameters and retries
    return healthcheck(timeout=timeout)


def index(
    business_id: str,
    content: bytes,
    mime_type: str,
    original_path: str,
    storage_path: Optional[str] = None,
) -> str:
    """Index a document by uploading it to an object storage and running the indexing pipeline.

    Returns the storage path used for the uploaded object.
    """
    s3 = get_s3_storage()

    file_extension = Path(original_path).suffix.lower()
    storage_path = storage_path or f"{uuid.uuid4()}{file_extension}"

    s3.upload_file(
        bucket=business_id,
        obj=storage_path,
        data=content,
        content_type=mime_type,
        metadata={"Original-Path": original_path},
    )

    byte_stream = ByteStream(data=content, mime_type=mime_type)
    pipeline = get_indexing_pipeline()
    pipeline.run(
        data={
            "converter": {
                "sources": [byte_stream],
                "meta": {"business_id": business_id, "file_path": storage_path},
            }
        }
    )

    return storage_path


def query(business_id: str, query_text: str, top_k: Optional[int] = None) -> Dict:
    """Run a query against the business index and return a structured result.

    Returns a dict with keys `answer` and `source_chunks`.
    """
    pipeline = get_query_pipeline()
    result = pipeline.run(
        data={
            "embedder": {"text": query_text},
            "retriever": {
                "filters": {"field": "meta.business_id", "operator": "==", "value": business_id},
                "top_k": top_k,
            },
            "prompt_builder": {"query": query_text},
        },
        include_outputs_from=["generator", "retriever"],
    )

    answer = result["generator"]["replies"][0]
    documents = result["retriever"]["documents"]

    source_chunks = [
        {
            "content": doc.content,
            "score": doc.score,
            "storage_path": doc.meta.get("file_path"),
        }
        for doc in documents
    ]

    return {"answer": answer, "source_chunks": source_chunks}


def list(business_id: str) -> List[Dict]:
    """List files stored for a business."""
    s3 = get_s3_storage()
    return s3.list_files(bucket=business_id)


def delete(business_id: str, storage_path: str) -> Dict[str, str]:
    """Delete an object from storage and its documents from the document store."""
    s3 = get_s3_storage()
    ds = get_document_store()

    try:
        s3.delete_file(bucket=business_id, obj=storage_path)
    except FileNotFoundError:
        return {"message": f"Document '{storage_path}' does not exist."}

    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.business_id", "operator": "==", "value": business_id},
            {"field": "meta.file_path", "operator": "==", "value": storage_path},
        ],
    }

    documents = ds.filter_documents(filters=filters)
    ids = [d.id for d in documents]
    if ids:
        ds.delete_documents(document_ids=ids)

    return {"message": f"Document '{storage_path}' deleted successfully."}


__all__ = ["connect", "index", "query", "list", "delete"]
