from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import MultiFileConverter
from haystack.components.preprocessors import DocumentPreprocessor
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.elasticsearch import (
    ElasticsearchEmbeddingRetriever,
)

from .config import settings
from .storage import _session, get_document_store
from .utils import NewlineNormalizer, get_document_embedder, get_generator, get_text_embedder


# TODO: Handle pipeline failures by removing remaining files and artifacts
class IndexingPipeline(Pipeline):
    """Pipeline for indexing documents: convert -> split -> embed -> store."""

    def __init__(self, warmup: bool = False):
        super().__init__()
        self.add_component("converter", MultiFileConverter())  # TODO: Handle "unclassified" files
        self.add_component("normalizer", NewlineNormalizer())
        self.add_component(
            "preprocessor",
            DocumentPreprocessor(
                split_by=settings.split_method,
                split_length=settings.chunk_size,
                split_overlap=settings.chunk_overlap,
            ),
        )
        self.add_component("embedder", get_document_embedder())
        self.add_component("writer", DocumentWriter(document_store=get_document_store()))

        self.connect("converter", "normalizer")
        self.connect("normalizer", "preprocessor")
        self.connect("preprocessor", "embedder")
        self.connect("embedder", "writer")

        if warmup:
            self.warm_up()


class QueryPipeline(Pipeline):
    """Pipeline for querying: retrieve relevant docs -> generate answer."""

    def __init__(self, warmup: bool = False):
        super().__init__()

        template = """
        Given the following RAG context information, answer the question.
        Don't use your own knowledge - only use the provided context.
        If the context is empty, say you don't know.
        If the answer isn't in the context, say you don't know.
        Be friendly, formal, and concise. No need for greetings.
        Respond in the same language as the question.

        <rag-context>
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}
        </rag-context>

        Question: {{query}}
        Answer:
        """

        self.add_component("embedder", get_text_embedder())
        self.add_component(
            "retriever", ElasticsearchEmbeddingRetriever(document_store=get_document_store())
        )
        self.add_component(
            "prompt_builder", PromptBuilder(template=template, required_variables="*")
        )
        self.add_component("generator", get_generator())

        self.connect("embedder.embedding", "retriever.query_embedding")
        self.connect("retriever.documents", "prompt_builder.documents")
        self.connect("prompt_builder.prompt", "generator.prompt")

        if warmup:
            self.warm_up()


def get_indexing_pipeline() -> IndexingPipeline:
    """Return the session's IndexingPipeline, creating and caching it on first call.

    The pipeline is created once and reused on every subsequent call.
    Haystack calls `warm_up()` automatically on the first `run()`.
    For eager warmup, call `needle.connect(warmup=True)` at startup instead.
    """
    if _session.indexing_pipeline is None:
        _session.indexing_pipeline = IndexingPipeline()
    return _session.indexing_pipeline  # type: ignore[return-value]


def get_query_pipeline() -> QueryPipeline:
    """Return the session's QueryPipeline, creating and caching it on first call.

    The pipeline is created once and reused on every subsequent call.
    Haystack calls `warm_up()` automatically on the first `run()`.
    For eager warmup, call `needle.connect(warmup=True)` at startup instead.
    """
    if _session.query_pipeline is None:
        _session.query_pipeline = QueryPipeline()
    return _session.query_pipeline  # type: ignore[return-value]
