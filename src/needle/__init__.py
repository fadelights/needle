"""
Needle
======

The needle package provides the core functionality for indexing documents
and semantically querying them. It also includes a FastAPI app that exposes
this functionality via a REST API.


Available functions
-------------------

import needle

needle.connect(...)
    Connect to external services (Elasticsearch, S3) and warm them up.
needle.index(...)
    Index a document: upload to object storage and run the indexing pipeline.
needle.query(...)
    Run a query against the business index and return a structured result.
needle.list_(...)
    List files stored for a business.
needle.delete(...)
    Delete an object from storage and its documents from the document store.
"""

from typing import Optional

from .config import settings
from .core import delete, index, list_, query
from .pipelines import IndexingPipeline, QueryPipeline
from .storage import _session, configure_session, healthcheck

# TODO: Registration and auth


def connect(
    *,
    es_host: Optional[str] = None,
    es_port: Optional[int] = None,
    es_scheme: Optional[str] = None,
    es_index: Optional[str] = None,
    s3_endpoint_url: Optional[str] = None,
    s3_access_key: Optional[str] = None,
    s3_secret_key: Optional[str] = None,
    s3_use_ssl: Optional[bool] = None,
    warmup: bool = True,
    timeout: int = 5,
) -> dict[str, bool]:
    """Connect to Elasticsearch and S3, then perform a health check.

    All connection parameters are optional and fall back to the values
    in `settings` (populated from environment variables / `.env`).
    Call this once at startup before using any other needle functions.

    Parameters
    ----------
    es_host:
        Elasticsearch hostname (default: `settings.es_host`).
    es_port:
        Elasticsearch port (default: `settings.es_port`).
    es_scheme:
        URL scheme for Elasticsearch, `"http"` or `"https"`
        (default: `settings.es_scheme`).
    es_index:
        Elasticsearch index name (default: `settings.es_index`).
    s3_endpoint_url:
        Full endpoint URL for the S3-compatible store, e.g.
        `"http://localhost:9000"` (default: built from
        `settings.minio_*`).
    s3_access_key:
        S3 access key (default: `settings.minio_root_user`).
    s3_secret_key:
        S3 secret key (default: `settings.minio_root_password`).
    s3_use_ssl:
        Whether to use SSL for S3 (default: `settings.minio_secure`).
    warmup:
        When `True` (default), both the indexing and query pipelines
        instantiate their components (e.g. loading ML models) at startup to minimize
        latency on the first request.  Set to `False` to defer this until the first
        call to `index()` or `query()`, which may be desirable in a serverless
        environment where cold-start time matters more than first-request latency.
    timeout:
        Reserved for future use (connection timeout in seconds).

    Returns
    -------
    dict[str, bool]
        `{"elasticsearch": True/False, "s3": True/False}` indicating
        whether each service is reachable.

    Examples
    --------
    Use defaults from .env / environment variables:

        import needle
        needle.connect()

    Override specific parameters at runtime:

        needle.connect(
            es_host="localhost",
            es_port=9200,
            s3_endpoint_url="http://localhost:9000",
            s3_access_key="user",
            s3_secret_key="password",
        )

    Skip pipeline warmup:

        needle.connect(warmup=False)
    """
    _es_host = es_host if es_host is not None else settings.es_host
    _es_port = es_port if es_port is not None else settings.es_port
    _es_scheme = es_scheme if es_scheme is not None else settings.es_scheme
    _es_index = es_index if es_index is not None else settings.es_index

    _s3_use_ssl = s3_use_ssl if s3_use_ssl is not None else settings.minio_secure
    _s3_endpoint = (
        s3_endpoint_url
        if s3_endpoint_url is not None
        else f"{'https' if _s3_use_ssl else 'http'}://{settings.minio_host}:{settings.minio_port}"
    )
    _s3_access_key = s3_access_key if s3_access_key is not None else settings.minio_root_user
    _s3_secret_key = s3_secret_key if s3_secret_key is not None else settings.minio_root_password

    configure_session(
        es_host=_es_host,
        es_port=_es_port,
        es_scheme=_es_scheme,
        es_index=_es_index,
        s3_endpoint_url=_s3_endpoint,
        s3_access_key=_s3_access_key,
        s3_secret_key=_s3_secret_key,
        s3_use_ssl=_s3_use_ssl,
    )

    if warmup:
        _session.indexing_pipeline = IndexingPipeline(warmup=True)
        _session.query_pipeline = QueryPipeline(warmup=True)
    else:
        # Clear any previously cached pipelines so stale instances are not reused
        # after a reconnect with different settings (e.g. a different ES index).
        _session.indexing_pipeline = None
        _session.query_pipeline = None

    return healthcheck(timeout=timeout)


__all__ = ["connect", "index", "query", "list_", "delete"]
