from haystack_integrations.document_stores.elasticsearch import (
    ElasticsearchDocumentStore,
)

from .config import settings

ES_MAPPING = {
    "dynamic": "strict",
    "properties": {
        "id": {"type": "keyword"},
        "file_path": {"type": "keyword"},
        "business_id": {"type": "keyword"},
        "source_id": {"type": "keyword"},
        "page_number": {"type": "integer"},
        "split_id": {"type": "integer"},
        "split_idx_start": {"type": "integer"},
        "_split_overlap": {
            "type": "nested",
            "properties": {
                "doc_id": {"type": "keyword"},
                "range": {"type": "integer"},
            },
        },
        "content": {"type": "text", "analyzer": "english"},
        "embedding": {
            "type": "dense_vector",
            "dims": settings.embedding_dim,
            "index": True,
            "similarity": "cosine",
        },
        "blob": {"type": "binary"},
        "score": {"type": "float"},
    },
}

document_store = ElasticsearchDocumentStore(
    hosts=f"{settings.es_scheme}://{settings.es_host}:{settings.es_port}",
    custom_mapping=ES_MAPPING,
    index=settings.es_index,
)
