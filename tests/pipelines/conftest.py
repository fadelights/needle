"""Shared fixtures for pipeline and document-store tests."""

import dataclasses
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from haystack import component
from haystack.dataclasses import Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# fmt: off
BUSINESS_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
OTHER_ID    = "ffffffff-gggg-hhhh-iiii-jjjjjjjjjjjj"
FILE_PATH   = "11111111-2222-3333-4444-555555555555.txt"
# fmt: on


@component
class DummyDocumentEmbedder:
    @component.output_types(documents=List[Document])
    def run(self, documents: List):
        return {
            "documents": [
                dataclasses.replace(doc, embedding=[0.1 * i] * 3) for i, doc in enumerate(documents)
            ]
        }


@component
class DummyTextEmbedder:
    @component.output_types(embedding=List[float])
    def run(self, text: str):
        return {"embedding": [0.1, 0.2, 0.3]}


@component
class DummyGenerator:
    @component.output_types(replies=List[str])
    def run(self, prompt: str):
        return {"replies": ["This is a serious generated answer."]}


def make_documents(**kwargs) -> Document:
    """Return a document with sensible defaults for pipeline tests."""
    defaults = dict(
        content="The quick brown fox jumps over the lazy dog.",
        meta={"bussiness_id": BUSINESS_ID, "file_path": FILE_PATH},
    )

    defaults.update(kwargs)
    return Document(**kwargs)


# ---------------------------------------------------------------------------
# In-memory document store & mocked pipeline components
# ---------------------------------------------------------------------------
@pytest.fixture()
def inmem_document_store():
    """
    InMemoryDocumentStore pre-populated with some documents.
    Embeddings are tiny float lists and no real model is needed.
    """
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    store = InMemoryDocumentStore()

    docs = [
        Document(
            content="ACME Corp was founded in 1999.",
            meta={"business_id": BUSINESS_ID, "file_path": "history.txt"},
            embedding=[0.1, 0.2, 0.3],
        ),
        Document(
            content="Our main product is the ACME Rocket.",
            meta={"business_id": BUSINESS_ID, "file_path": "products.txt"},
            embedding=[0.4, 0.5, 0.6],
        ),
        Document(
            content="Totally unrelated document from another tenant.",
            meta={"business_id": OTHER_ID, "file_path": "other.txt"},
            embedding=[0.7, 0.8, 0.9],
        ),
    ]

    store.write_documents(docs)
    return store


@pytest.fixture()
def mock_document_embedder():
    return DummyDocumentEmbedder()


@pytest.fixture()
def mock_text_embedder():
    return DummyTextEmbedder()


@pytest.fixture()
def mock_generator():
    return DummyGenerator()
