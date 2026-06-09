from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

import needle.core as core


def test_index_no_storage_path(core_mocks):
    mock_storage, _, mock_indexing_pipeline, _ = core_mocks

    business_id = "acme"
    content = b"hello world"
    mime_type = "text/markdown"
    original_path = "document.MD"

    result = core.index(
        business_id=business_id,
        content=content,
        mime_type=mime_type,
        original_path=original_path,
    )
    assert isinstance(result, str)
    assert result.endswith(".md")  # extension should be lowercased

    stem = result.removesuffix(".md")
    UUID(stem)  # raises ValueError if not a valid UUID

    mock_storage.upload_file.assert_called_once_with(
        bucket=business_id,
        obj=result,
        data=content,
        content_type=mime_type,
        metadata={"Original-Path": original_path},
    )

    mock_indexing_pipeline.run.assert_called_once()
    call_args = mock_indexing_pipeline.run.call_args.kwargs["data"]
    assert call_args["converter"]["meta"]["business_id"] == business_id
    assert call_args["converter"]["meta"]["file_path"] == result


def test_index_with_storage_path(core_mocks):
    """When storage_path is provided, it should be used as-is."""
    mock_storage, _, mock_indexing_pipeline, _ = core_mocks

    business_id = "acme"
    content = b"hello world"
    mime_type = "text/markdown"
    original_path = "notes.PDF"
    storage_path = "stored.PDF"

    result = core.index(
        business_id=business_id,
        content=content,
        mime_type=mime_type,
        original_path=original_path,
        storage_path=storage_path,
    )

    assert result.endswith(".pdf")

    # Verify the generated path was used for both upload and pipeline meta
    mock_storage.upload_file.assert_called_once()
    assert mock_storage.upload_file.call_args.kwargs["obj"] == result
    assert mock_storage.upload_file.call_args.kwargs["metadata"]["Original-Path"] == original_path

    mock_indexing_pipeline.run.assert_called_once()
    pipeline_data = mock_indexing_pipeline.run.call_args.kwargs["data"]
    assert pipeline_data["converter"]["meta"]["file_path"] == result


def test_delete(core_mocks):
    mock_storage, mock_document_store, _, _ = core_mocks

    business_id = "acme"
    storage_path = "stored.txt"

    mock_document_store.filter_documents.return_value = [Mock(id="doc1"), Mock(id="doc2")]

    result = core.delete(business_id=business_id, storage_path=storage_path)

    assert result == {"message": f"Document '{storage_path}' deleted successfully."}
    mock_storage.delete_file.assert_called_once_with(bucket=business_id, obj=storage_path)
    mock_document_store.filter_documents.assert_called_once()
    mock_document_store.delete_documents.assert_called_once_with(document_ids=["doc1", "doc2"])


def test_delete_filters():
    store = InMemoryDocumentStore()

    docs = [
        Document(content="acme report", meta={"business_id": "acme", "file_path": "report.txt"}),
        Document(content="acme notes", meta={"business_id": "acme", "file_path": "notes.txt"}),
        Document(content="other report", meta={"business_id": "other", "file_path": "report.txt"}),
    ]
    store.write_documents(docs)

    mock_storage = Mock()
    with (
        patch("needle.core.get_s3_storage", return_value=mock_storage),
        patch("needle.core.get_document_store", return_value=store),
    ):
        core.delete(business_id="acme", storage_path="report.txt")

    remaining = store.filter_documents()  # running with no filter returns all documents
    assert len(remaining) == 2

    remaining_pairs = {(d.meta["business_id"], d.meta["file_path"]) for d in remaining}
    assert ("acme", "report.txt") not in remaining_pairs
    assert ("acme", "notes.txt") in remaining_pairs
    assert ("other", "report.txt") in remaining_pairs


def test_list_(core_mocks):
    mock_storage, _, _, _ = core_mocks

    business_id = "acme"
    expected_files = [{"key": "a.txt"}]
    mock_storage.list_files.return_value = expected_files

    result = core.list_(business_id=business_id)

    assert result == expected_files
    mock_storage.list_files.assert_called_once_with(bucket=business_id)


def test_get_content(core_mocks):
    mock_storage, _, _, _ = core_mocks

    business_id = "acme"
    storage_path = "a.txt"
    content = "hello"
    mock_storage.get_file.return_value = content.encode("utf-8")

    result = core.get_content(business_id=business_id, storage_path=storage_path)

    assert result == "hello"
    mock_storage.get_file.assert_called_once_with(bucket=business_id, obj=storage_path)


def test_update_content(core_mocks):
    mock_storage, mock_document_store, mock_indexing_pipeline, _ = core_mocks

    business_id = "acme"
    original_path = "a.txt"
    storage_path = "b.txt"
    new_content = "updated"

    mock_storage.list_files.return_value = [
        {"key": storage_path, "metadata": {"original-path": original_path}}
    ]
    mock_document_store.filter_documents.return_value = []  # simply need an iterable

    result = core.update_content(
        business_id=business_id,
        storage_path=storage_path,
        content=new_content,
    )

    assert result == {"message": f"File '{storage_path}' updated and re-indexed successfully."}
    mock_storage.list_files.assert_called_once_with(bucket=business_id)
    mock_storage.delete_file.assert_called_once_with(bucket=business_id, obj=storage_path)
    mock_storage.upload_file.assert_called_once()
    mock_indexing_pipeline.run.assert_called_once()


def test_update_content_unsupported_extension():
    with pytest.raises(ValueError, match="Only .* files can be modified"):
        core.update_content(
            business_id="acme",
            storage_path="book.pdf",
            content="hello",
        )


def test_update_content_missing_file(core_mocks):
    mock_storage, _, _, _ = core_mocks

    business_id = "acme"
    storage_path = "missing.txt"
    mock_storage.list_files.return_value = []

    with pytest.raises(FileNotFoundError, match=f"File '{storage_path}' not found."):
        core.update_content(
            business_id=business_id,
            storage_path=storage_path,
            content="hello",
        )


def test_query(core_mocks):
    _, _, _, mock_query_pipeline = core_mocks

    business_id = "acme"
    query_text = "hello"
    mock_query_pipeline.run.return_value = {
        "generator": {"replies": ["answer"]},
        "retriever": {"documents": [Mock(content="text", score=0.9, meta={"file_path": "a.txt"})]},
    }

    result = core.query(business_id=business_id, text=query_text, top_k=5)

    assert result == {
        "answer": "answer",
        "source_chunks": [{"content": "text", "score": 0.9, "storage_path": "a.txt"}],
    }
    mock_query_pipeline.run.assert_called_once()
