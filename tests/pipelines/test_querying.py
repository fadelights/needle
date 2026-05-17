from unittest.mock import MagicMock, patch

import pytest
from haystack import Pipeline
from haystack.dataclasses import Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# fmt: off
BUSINESS_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
OTHER_ID    = "ffffffff-gggg-hhhh-iiii-jjjjjjjjjjjj"
FILE_PATH   = "11111111-2222-3333-4444-555555555555.txt"
# fmt: on


def _run_pipeline(
    pipeline: Pipeline, query: str, *, business_id: str = BUSINESS_ID, top_k: int = 3
):
    """Convenience wrapper to run the pipeline with a query."""
    if not issubclass(pipeline.__class__, Pipeline):
        raise ValueError(f"{pipeline.__class__!r} is not a proper Haystack Pipeline.")

    return pipeline.run(
        data={
            "embedder": {"text": query},
            "retriever": {
                "filters": {
                    "field": "meta.business_id",
                    "operator": "==",
                    "value": business_id,
                },
                "top_k": top_k,
            },
            "prompt_builder": {"query": query},
        },
        include_outputs_from=["generator", "retriever"],
    )


@pytest.fixture()
def query_pipeline(inmem_document_store, mock_text_embedder, mock_generator):
    """QueryPipeline utilizing an InMemoryDocumentStore and a
    fake embedder and generator."""
    from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

    with (
        patch("needle.pipelines.get_text_embedder", return_value=mock_text_embedder),
        patch("needle.pipelines.get_generator", return_value=mock_generator),
        patch("needle.pipelines.document_store", inmem_document_store),
        patch(
            "needle.pipelines.ElasticsearchEmbeddingRetriever",
            lambda **kwargs: InMemoryEmbeddingRetriever(**kwargs),
        ),
    ):
        from needle.pipelines import QueryPipeline

        yield QueryPipeline()


# ---------------------------------------------------------------------------


class TestQueryPipeline:
    def test_returns_answer(self, query_pipeline):
        result = _run_pipeline(query_pipeline, "What does ACME make?")
        answer = result["generator"]["replies"][0]

        assert isinstance(answer, str)
        assert answer.strip()

    def test_returns_sources(self, query_pipeline):
        result = _run_pipeline(query_pipeline, "What does ACME make?")
        documents = result["retriever"]["documents"]

        assert isinstance(documents, list)
        assert all(isinstance(d, Document) for d in documents)

    def test_top_k_limits_results(self, query_pipeline):
        result = _run_pipeline(query_pipeline, "What does ACME make?", top_k=1)

        assert len(result["retriever"]["documents"]) <= 1

    def test_tenant_isolation(self, query_pipeline):
        result = _run_pipeline(query_pipeline, "What does ACME make?", business_id=BUSINESS_ID)

        for doc in result["retriever"]["documents"]:
            assert (
                doc.meta.get("business_id") == BUSINESS_ID
            ), f"Document leakage from another tenant: {doc.meta}"

        result = _run_pipeline(query_pipeline, "Tell me something.", business_id=OTHER_ID)

        for doc in result["retriever"]["documents"]:
            assert (
                doc.meta.get("business_id") == OTHER_ID
            ), f"Document leakage from another tenant: {doc.meta}"

    def test_generator_prompt_contains_query(self, query_pipeline, mock_generator):
        query = "What are ACME's Rockets made of?"
        _run_pipeline(query_pipeline, query)

        assert query in mock_generator.last_prompt

        # The prompt template injects document content before the question,
        # so the prompt must be longer than the query alone.
        assert len(mock_generator.last_prompt.strip()) > len(query)

    def test_generator_prompt_contains_context(self, query_pipeline, mock_generator):
        result = _run_pipeline(query_pipeline, "What does ACME make?")
        documents = result["retriever"]["documents"]

        assert documents, "No documents were retrieved."

        for document in documents:
            assert (
                document.content in mock_generator.last_prompt
            ), f"Document content not found in prompt: {document.content!r}"
