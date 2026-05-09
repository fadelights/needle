from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import MultiFileConverter
from haystack.components.preprocessors import DocumentPreprocessor
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.elasticsearch import (
    ElasticsearchEmbeddingRetriever,
)

from .config import settings
from .storage import document_store
from .utils import (
    NewlineNormalizer,
    get_document_embedder,
    get_generator,
    get_text_embedder,
)


class IndexingPipeline(Pipeline):
    """Pipeline for indexing documents: convert -> split -> embed -> store."""

    def __init__(self):
        super().__init__()
        self.add_component(
            "converter", MultiFileConverter()
        )  # TODO: Handle "unclassified" files
        self.add_component("normalizer", NewlineNormalizer())
        self.add_component(
            "preprocessor",
            DocumentPreprocessor(
                split_by="sentence",
                split_length=settings.chunk_size,
                split_overlap=settings.chunk_overlap,
            ),
        )
        self.add_component("embedder", get_document_embedder())
        self.add_component("writer", DocumentWriter(document_store=document_store))

        self.connect("converter", "normalizer")
        self.connect("normalizer", "preprocessor")
        self.connect("preprocessor", "embedder")
        self.connect("embedder", "writer")


class QueryPipeline(Pipeline):
    """Pipeline for querying: retrieve relevant docs -> generate answer."""

    def __init__(self):
        super().__init__()
        # TODO: The agent will hallucinate if there are no docs
        template = """
        Given the following information, answer the question.
        Don't use your own knowledge - only use the provided documents.
        If you don't know the answer, say you don't know.
        Be friendly, but concise.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{query}}
        Answer:
        """
        self.add_component("embedder", get_text_embedder())
        self.add_component(
            "retriever", ElasticsearchEmbeddingRetriever(document_store=document_store)
        )
        self.add_component(
            "prompt_builder", PromptBuilder(template=template, required_variables="*")
        )
        self.add_component("generator", get_generator())

        self.connect("embedder.embedding", "retriever.query_embedding")
        self.connect("retriever.documents", "prompt_builder.documents")
        self.connect("prompt_builder.prompt", "generator.prompt")
