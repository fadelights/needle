import io
from unittest.mock import MagicMock, patch

import pytest
from conftest import BUSINESS_ID, FILE_PATH, OTHER_ID
from haystack import Pipeline
from haystack.dataclasses import ByteStream, Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bytestream(text: str, mime: str):
    """Mocks the given `text` as a file's bytestream."""
    return ByteStream(
        data=text.encode(),  # this is the representation we get when using FastAPI's `file.read()`
        mime_type=mime,
    )


def _run_pipeline(pipeline, text: str, mime: str):
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
                "meta": {"business_id": BUSINESS_ID, "file_path": FILE_PATH},
            }
        }
    )


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
    def test_plain_text_document_is_written():
        pass

    def test_metadata_is_preserved():
        pass

    def test_embeddings_are_set():
        pass

    def long_document_is_split():
        pass

    def newlines_are_normalized():
        pass

    def test_multiple_files_are_isolated():
        pass
