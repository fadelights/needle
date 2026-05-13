"""Shared fixtures for pipeline and document-store tests."""

from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack_integrations.document_stores.elasticsearch import (
    ElasticsearchDocumentStore,
)

from needle.config import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# fmt: off
BUSINESS_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
OTHER_ID    = "ffffffff-gggg-hhhh-iiii-jjjjjjjjjjjj"
FILE_PATH   = "11111111-2222-3333-4444-555555555555.txt"
# fmt: on


def make_documents(**kwargs) -> Document:
    """Return a document with sensible defaults for pipeline tests."""
    defaults = dict(
        content="The quick brown fox jumps over the lazy dog.",
        meta={"bussiness_id": BUSINESS_ID, "file_path": FILE_PATH},
    )

    defaults.update(kwargs)
    return Document(**kwargs)


# ---------------------------------------------------------------------------
# Elasticsearch document store
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def es_document_store():
    """
    Real ElasticsearchDocumentStore pointed at the configured ES instance.

    Expects ES to be reachable at the configured ENV host. If not, will
    use whatever defaults set for the app. It will use a dummy index
    for performing the tests.

    The index is dropped after the session so tests stay idempotent.
    """
    index = "test"
    from needle.storage import ES_MAPPING

    store = ElasticsearchDocumentStore(
        hosts=f"{settings.es_scheme}://{settings.es_host}:{settings.es_port}",
        custom_mapping=ES_MAPPING,
        index=index,
    )

    yield store

    # Teardown
    try:
        if store._client is not None:
            store._client.indices.delete(index=index, ignore_unavailable=True)
    except Exception as exc:
        raise Exception(f"ES teardown failed due to the following reason: {exc}")


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
    """
    Uses a fake embedder that adds a dummy embedding to every document.
    """

    def _fake_run(documents):
        for i, doc in enumerate(documents, start=1):
            doc.embedding = [0.1 * i] * 3
        return {"documents": documents}

    embedder = MagicMock()
    embedder.run.side_effect = _fake_run

    # Haystack inspects these at pipeline connect-time
    embedder.__haystack_input__ = {"documents": MagicMock()}
    embedder.__haystack_output__ = {"documents": MagicMock()}

    return embedder


@pytest.fixture()
def mock_text_embedder():
    """Fake text embedder that always returns the same query vector."""

    def _fake_run(text):
        return {"embedding": [0.1, 0.2, 0.3]}

    embedder = MagicMock()
    embedder.run.side_effect = _fake_run
    embedder.__haystack_input__ = {"text": MagicMock()}
    embedder.__haystack_output__ = {"embedding": MagicMock()}

    return embedder


@pytest.fixture()
def mock_generator():
    """Fake LLM generator that echoes back a canned answer."""

    def _fake_run(prompt):
        return {"replies": ["This is a generated answer."]}

    generator = MagicMock()
    generator.run.side_effect = _fake_run
    generator.__haystack_input__ = {"prompt": MagicMock()}
    generator.__haystack_output__ = {"replies": MagicMock()}

    return generator
