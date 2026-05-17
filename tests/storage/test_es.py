"""
Integration tests for the Elasticsearch document store.

These tests require a live ES node either specified by
your ENV settings or by whatever defaults the package uses.

Run "only" them with:

```
pytest -m integration
```

Skip them by running:
```
pytest -m 'not integration'
```
"""

import pytest

pytestmark = pytest.mark.integration


def test_index_created_with_correct_mapping(es_document_store):
    es_document_store._ensure_initialized()

    assert es_document_store.client.indices.exists(index=es_document_store._index)


def test_mapping_rejects_extra_fields(es_document_store):
    """dynamic=strict means ES must reject documents with unknown fields."""
    from elasticsearch import BadRequestError

    es_document_store._ensure_initialized()
    client = es_document_store.client
    index = es_document_store._index

    with pytest.raises(BadRequestError):
        client.index(
            index=index,
            document={
                "id": "bad-doc",
                "content": "malicious",
                "unknown_field": "this should be rejected.",
            },
            refresh=True,
        )


def test_embedding_field_is_dense_vector(es_document_store):
    es_document_store._ensure_initialized()
    client = es_document_store.client
    index = es_document_store._index
    mapping = client.indices.get_mapping(index=index)
    props = mapping[index]["mappings"]["properties"]

    assert props["embedding"]["type"] == "dense_vector"
