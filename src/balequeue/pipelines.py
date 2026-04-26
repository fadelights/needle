from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator, HuggingFaceLocalGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.components.retrievers.elasticsearch import (
    ElasticsearchEmbeddingRetriever,
)

from .config import settings
from .storage import document_store


class IndexingPipeline(Pipeline):
    """Pipeline for indexing documents: convert -> split -> embed -> store."""

    def __init__(self):
        super().__init__()
        self.add_component("converter", TextFileToDocument())
        self.add_component(
            "splitter",
            DocumentSplitter(
                split_by="sentence",
                split_length=settings.chunk_size,
                split_overlap=settings.chunk_overlap,
            )
        )
        self.add_component(
            "embedder",
            OpenAIDocumentEmbedder(
                api_key=Secret.from_token(settings.openai_api_key),
                model=settings.embedding_model,
            ),
        )
        self.add_component("writer", DocumentWriter(document_store=document_store))

        self.connect("converter", "splitter")
        self.connect("splitter", "embedder")
        self.connect("embedder", "writer")


class QueryPipeline(Pipeline):
    """Pipeline for querying: retrieve relevant docs -> generate answer."""

    def __init__(self):
        super().__init__()
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
        self.add_component(
            "embedder",
            OpenAITextEmbedder(
                api_key=Secret.from_token(settings.openai_api_key),
                model=settings.embedding_model,
            ),
        )
        self.add_component(
            "retriever", ElasticsearchEmbeddingRetriever(document_store=document_store)
        )
        self.add_component(
            "prompt_builder", PromptBuilder(template=template, required_variables="*")
        )
        self.add_component(
            "generator",
            OpenAIGenerator(
                api_key=Secret.from_token(settings.openai_api_key),
                model=settings.generator_model,
            ),
        )

        self.connect("embedder.embedding", "retriever.query_embedding")
        self.connect("retriever.documents", "prompt_builder.documents")
        self.connect("prompt_builder.prompt", "generator.prompt")
