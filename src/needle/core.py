from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional

from haystack.dataclasses import ByteStream

from .config import settings
from .pipelines import get_indexing_pipeline, get_query_pipeline
from .storage import get_document_store, get_s3_storage


def index(
    business_id: str,
    content: bytes,
    mime_type: str,
    original_path: str,
    storage_path: Optional[str] = None,
) -> str:
    """Upload content to storage and index it through the pipeline."""
    file_extension = Path(original_path).suffix.lower()
    storage_path = storage_path.lower() if storage_path else f"{uuid.uuid4()}{file_extension}"

    s3 = get_s3_storage()
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


def delete(business_id: str, storage_path: str) -> Dict[str, str]:
    """Delete an object from storage and remove its indexed documents."""
    s3 = get_s3_storage()
    ds = get_document_store()

    s3.delete_file(bucket=business_id, obj=storage_path)

    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.business_id", "operator": "==", "value": business_id},
            {"field": "meta.file_path", "operator": "==", "value": storage_path},
        ],
    }

    documents = ds.filter_documents(filters=filters)
    ids = [document.id for document in documents]
    if ids:
        ds.delete_documents(document_ids=ids)

    return {"message": f"Document '{storage_path}' deleted successfully."}


def list_(business_id: str) -> List[Dict[str, object]]:
    """List files stored for a business."""
    s3 = get_s3_storage()
    return s3.list_files(bucket=business_id)


def get_content(business_id: str, storage_path: str) -> str:
    """Return UTF-8 text content for a stored file."""
    s3 = get_s3_storage()
    content_bytes = s3.get_file(bucket=business_id, obj=storage_path)
    return content_bytes.decode("utf-8")


def update_content(business_id: str, storage_path: str, content: str) -> Dict[str, str]:
    """Update a stored text file and re-index it."""
    extension = Path(storage_path).suffix.lower()
    allowed = [".txt", ".md"]
    if extension not in allowed:
        raise ValueError(f"Only {', '.join(allowed)} files can be modified.")

    s3 = get_s3_storage()
    files = s3.list_files(bucket=business_id)
    existing = next(
        (
            (file["key"], file["metadata"].get("original-path"))
            for file in files
            if file["key"] == storage_path
        ),
        None,
    )

    if not existing:
        raise FileNotFoundError(f"File '{storage_path}' not found.")

    _, original_path = existing

    delete(business_id=business_id, storage_path=storage_path)

    mime_mapping = {".txt": "text/plain", ".md": "text/markdown"}
    mime_type = mime_mapping.get(extension, "text/plain")
    index(
        business_id=business_id,
        content=content.encode("utf-8"),
        mime_type=mime_type,
        original_path=original_path,
        storage_path=storage_path,
    )
    return {"message": f"File '{storage_path}' updated and re-indexed successfully."}


def query(
    business_id: str,
    text: str,
    top_k: Optional[int] = None,
    generate_response: Optional[bool] = None,
) -> Dict[str, object]:
    """Run a query pipeline and return answer with source chunk metadata.

    Parameters
    ----------
    top_k: Optional[int]
        Number of top relevant chunks to retrieve.
        Overrides the default ENV settings if provided.
    generate_response: Optional[bool]
        Whether to generate a response using the LLM.
        Overrides the default ENV settings if provided.

    Notes
    -----
    Normally, `generate_response` should be `True` (default behavior if
    nothing specified in function call or ENV). Setting `generate_response` to
    `False` will skip the LLM step and return only retrieved chunks, which
    can be useful when using Needle in a bigger system and you want to handle
    response generation separately.
    """
    pipeline = get_query_pipeline()

    top_k = top_k or settings.top_k
    generate_response = (
        generate_response if generate_response is not None else settings.generate_response
    )

    retriever_filters = {"field": "meta.business_id", "operator": "==", "value": business_id}

    if not generate_response:
        embedding = pipeline.get_component("embedder").run(text=text)["embedding"]
        documents = pipeline.get_component("retriever").run(
            query_embedding=embedding,
            filters=retriever_filters,
            top_k=top_k,
        )["documents"]
        answer = None
    else:
        result = pipeline.run(
            data={
                "embedder": {"text": text},
                "retriever": {
                    "filters": retriever_filters,
                    "top_k": top_k,
                },
                "prompt_builder": {"query": text},
            },
            include_outputs_from=["generator", "retriever"],
        )
        documents = result["retriever"]["documents"]
        answer = result["generator"]["replies"][0]

    source_chunks: List[Dict[str, object]] = [
        {
            "content": doc.content,
            "score": doc.score,
            "storage_path": doc.meta.get("file_path"),
        }
        for doc in documents
    ]

    return {"answer": answer, "source_chunks": source_chunks}
