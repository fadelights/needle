from haystack.components.embedders import (
    OpenAIDocumentEmbedder,
    OpenAITextEmbedder,
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import HuggingFaceLocalGenerator, OpenAIGenerator
from haystack.utils import ComponentDevice, Secret

from .config import settings


def _get_device():
    device = ComponentDevice.from_str("cuda:0")
    return ComponentDevice.resolve_device(device)


def _get_document_embedder():
    """Create document embedder based on provider setting."""
    if settings.embedding_provider.lower() == "huggingface":
        return SentenceTransformersDocumentEmbedder(
            model=settings.embedding_model,
            device=_get_device(),
        )
    elif settings.embedding_provider.lower() == "openai":
        return OpenAIDocumentEmbedder(
            api_key=Secret.from_token(settings.openai_api_key),
            model=settings.embedding_model,
        )


def _get_text_embedder():
    """Create text embedder based on provider setting."""
    if settings.embedding_provider.lower() == "huggingface":
        return SentenceTransformersTextEmbedder(
            model=settings.embedding_model,
            device=_get_device(),
        )
    elif settings.embedding_provider.lower() == "openai":
        return OpenAITextEmbedder(
            api_key=Secret.from_token(settings.openai_api_key),
            model=settings.embedding_model,
        )


def _get_generator():
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
            model=settings.generator_model,
        )
