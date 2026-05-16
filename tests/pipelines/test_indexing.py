import io
import os
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from haystack import Pipeline
from haystack.dataclasses import ByteStream, Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# fmt: off
BUSINESS_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
OTHER_ID    = "ffffffff-gggg-hhhh-iiii-jjjjjjjjjjjj"
FILE_PATH   = "11111111-2222-3333-4444-555555555555.txt"
# fmt: on


def _bytestream(text: str, mime: str):
    """Mocks the given `text` as a file's bytestream."""
    return ByteStream(
        data=text.encode(),  # this is the representation we get when using FastAPI's `file.read()`
        mime_type=mime,
    )


def _run_pipeline(
    pipeline: Pipeline,
    text: str,
    *,
    mime: str = "text/plain",
    file_path: str = FILE_PATH,
    bussines_id: str = BUSINESS_ID,
):
    """Convenience wrapper to run the pipeline with a text **file**.

    Note that even though it's receiving plain text,
    the `text` input will be treated as a file.
    """
    if not issubclass(pipeline.__class__, Pipeline):
        raise ValueError(f"{pipeline} is not a proper Haystack Pipeline.")

    pipeline.run(
        data={
            "converter": {
                "sources": [_bytestream(text, mime)],
                "meta": {"business_id": bussines_id, "file_path": file_path},
            }
        }
    )


def _filter_document(store, business_id, file_path) -> List[Document]:
    "Return all documents in `store` that match the given `business_id` and `file_path`."
    filters = {
        "operator": "AND",
        "conditions": [
            {
                "field": "meta.business_id",
                "operator": "==",
                "value": business_id,
            },
            {
                "field": "meta.file_path",
                "operator": "==",
                "value": file_path,
            },
        ],
    }
    return store.filter_documents(filters)


@pytest.fixture()
def indexing_pipeline(inmem_document_store, mock_document_embedder):
    """
    IndexingPipeline utilizing an InMemoryDocumentStore and a
    fake embedder.
    """
    with (
        patch("needle.pipelines.get_document_embedder", return_value=mock_document_embedder),
        patch("needle.pipelines.document_store", inmem_document_store),
    ):
        from needle.pipelines import IndexingPipeline

        yield IndexingPipeline()


# ---------------------------------------------------------------------------


class TestIndexingPipeline:
    def test_plain_text_document_is_written(self, indexing_pipeline, inmem_document_store):
        initial_count = inmem_document_store.count_documents()

        _run_pipeline(indexing_pipeline, "ACME Corp makes great rockets.")

        assert inmem_document_store.count_documents() == initial_count + 1

    def test_metadata_is_preserved(self, indexing_pipeline, inmem_document_store):
        "Custom metadata must survive the full pipeline."
        cases = [
            ("1", "/path/to/file-a1.txt", "ACME Corp makes great rockets."),
            ("1", "/path/to/file-a2.txt", "ACME Corp makes superb spaceships."),
            ("13", "/path/to/file-b.txt", "ACNE is dedicated to the growth of human-kind."),
            ("135", "/path/to/file-c.txt", "ACOE products are innovative and reliable."),
        ]

        for business_id, file_path, statement in cases:
            _run_pipeline(
                indexing_pipeline, statement, file_path=file_path, bussines_id=business_id
            )

        for business_id, file_path, _ in cases:
            documents = _filter_document(
                inmem_document_store,
                business_id,
                os.path.basename(
                    file_path
                ),  # Remember that our pipeline doesn't store the full path
            )
            assert len(documents) == 1, f"Expected exactly one document for {file_path!r}"
            assert documents[0].meta["business_id"] == business_id

    def test_embeddings_are_set(self, indexing_pipeline, inmem_document_store):
        pass

    def test_long_document_is_split(self):
        pass

    def test_newlines_are_normalized(self):
        pass

    def test_multiple_files_are_isolated(self):
        pass
