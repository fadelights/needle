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
        raise ValueError(f"{pipeline} is not a proper Haystack Pipeline.")

    return pipeline.run(
        data={
            "embedder": {"text": query},
            "retriever": {
                "filters": {
                    "field": "meta.business_id",
                    "opertaor": "==",
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
            lambda documnet_store: InMemoryEmbeddingRetriever(document_store=documnet_store),
        ),
    ):
        from needle.pipelines import QueryPipeline

        yield QueryPipeline


# ---------------------------------------------------------------------------


class TestQueryPipeline:
    def test_returns_answer(self, query_pipeline):
        pass

    def test_returns_sources(self, query_pipeline):
        pass

    def test_top_k_limits_results(self, query_pipeline):
        pass

    def test_tenant_isolation(self, query_pipeline):
        pass

    def test_tenant_sees_own_docs(self, query_pipeline):
        pass

    def test_generator_prompt_contains_query(self, query_pipeline):
        pass

    def test_generator_prompt_contains_retrieved_docs(self, query_pipeline):
        pass
