from haystack_integrations.document_stores.elasticsearch import (
    ElasticsearchDocumentStore,
)

from .config import settings

ES_MAPPING = {
    "dynamic": "strict",
    "properties": {
        "business_id": {"type": "keyword"},
        "doc_id": {"type": "keyword"},
        "chunk_index": {"type": "integer"},
        "file_path": {"type": "keyword"},
        "content": {"type": "text", "analyzer": "english"},
        "embedding": {
            "type": "dense_vector",
            "dims": settings.embedding_dim,
            "index": True,
            "similarity": "cosine",
        },
        "metadata": {
            "type": "object",
            "properties": {
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
            },
        },
    },
}

document_store = ElasticsearchDocumentStore(
    hosts=f"{settings.es_scheme}://{settings.es_host}:{settings.es_port}",
    custom_mapping=ES_MAPPING,
    index=settings.es_index,
)
