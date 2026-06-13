import dataclasses
from typing import List

from haystack import Document, component
from haystack.components.embedders import (
    OpenAIDocumentEmbedder,
    OpenAITextEmbedder,
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import HuggingFaceLocalGenerator, OpenAIGenerator
from haystack.utils import ComponentDevice, Secret
from haystack_integrations.components.embedders.ollama import (
    OllamaDocumentEmbedder,
    OllamaTextEmbedder,
)
from haystack_integrations.components.generators.ollama import OllamaGenerator

from .config import settings


@component
class NewlineNormalizer:
    """Preprocessor component that normalizes newlines
    to `\\n` in document content."""

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> List[Document]:
        return {
            "documents": [
                (
                    dataclasses.replace(document, content=document.content.replace("\r\n", "\n"))
                    if document.content
                    else document
                )
                for document in documents
            ]
        }


def _get_device() -> ComponentDevice:
    """Resolve a device string specified via the GENERATOR_DEVICE environment setting,
    falling back to CPU when the preferred device is unavailable (e.g. no GPU present)."""
    try:
        device = ComponentDevice.from_str(settings.generator_device)
        return ComponentDevice.resolve_device(device)
    except Exception:
        return ComponentDevice.from_str("cpu")


def get_document_embedder():
    """Create document embedder based on provider setting."""
    if settings.embedding_provider.lower() == "huggingface":
        return SentenceTransformersDocumentEmbedder(
            model=settings.embedding_model,
            device=_get_device(),
        )
    elif settings.embedding_provider.lower() == "openai":
        return OpenAIDocumentEmbedder(
            api_key=Secret.from_token(settings.openai_api_key),
            api_base_url=settings.openai_base_url,
            model=settings.embedding_model,
        )
    elif settings.embedding_provider.lower() == "ollama":
        return OllamaDocumentEmbedder(
            model=settings.embedding_model,
            url=settings.ollama_base_url,
        )


def get_text_embedder():
    """Create text embedder based on provider setting."""
    if settings.embedding_provider.lower() == "huggingface":
        return SentenceTransformersTextEmbedder(
            model=settings.embedding_model,
            device=_get_device(),
        )
    elif settings.embedding_provider.lower() == "openai":
        return OpenAITextEmbedder(
            api_key=Secret.from_token(settings.openai_api_key),
            api_base_url=settings.openai_base_url,
            model=settings.embedding_model,
        )
    elif settings.embedding_provider.lower() == "ollama":
        return OllamaTextEmbedder(
            model=settings.embedding_model,
            url=settings.ollama_base_url,
        )


def get_generator():
    """Create generator based on provider setting."""
    if settings.generator_provider.lower() == "huggingface":
        return HuggingFaceLocalGenerator(
            model=settings.generator_model,
            task="text-generation",
            device=_get_device(),
            token=Secret.from_token(settings.hf_api_token),
            generation_kwargs={"max_new_tokens": settings.max_new_tokens},
        )
    elif settings.generator_provider.lower() == "openai":
        return OpenAIGenerator(
            api_key=Secret.from_token(settings.openai_api_key),
            api_base_url=settings.openai_base_url,
            model=settings.generator_model,
        )
    elif settings.generator_provider.lower() == "ollama":
        return OllamaGenerator(
            model=settings.generator_model,
            url=settings.ollama_base_url,
            generation_kwargs={"num_predict": settings.max_new_tokens},
        )
