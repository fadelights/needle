from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
from haystack import Pipeline
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

from .config import settings


class _Session:
    """Holds the active document-store, S3 storage, and pipeline instances."""

    def __init__(self) -> None:
        self._document_store: Optional[ElasticsearchDocumentStore] = None
        self._s3_storage: Optional[S3Storage] = None
        self.indexing_pipeline: Optional[Pipeline] = None
        self.query_pipeline: Optional[Pipeline] = None

    @property
    def document_store(self) -> ElasticsearchDocumentStore:
        if self._document_store is None:
            self._document_store = _build_document_store(
                host=settings.es_host,
                port=settings.es_port,
                scheme=settings.es_scheme,
                index=settings.es_index,
            )
        return self._document_store

    @document_store.setter
    def document_store(self, value: ElasticsearchDocumentStore) -> None:
        self._document_store = value

    @property
    def s3_storage(self) -> "S3Storage":
        if self._s3_storage is None:
            self._s3_storage = S3Storage(
                endpoint_url=f"{'https' if settings.minio_secure else 'http'}://{settings.minio_host}:{settings.minio_port}",
                access_key=settings.minio_root_user,
                secret_key=settings.minio_root_password,
                use_ssl=settings.minio_secure,
            )
        return self._s3_storage

    @s3_storage.setter
    def s3_storage(self, value: "S3Storage") -> None:
        self._s3_storage = value


_session = _Session()


def _build_document_store(
    host: str,
    port: int,
    scheme: str = "http",
    index: str = "documents",
    embedding_dim: Optional[int] = None,
    language: Optional[str] = None,
) -> ElasticsearchDocumentStore:
    """Construct an ElasticsearchDocumentStore with the needle mapping."""
    if embedding_dim is None:
        embedding_dim = settings.embedding_dim
    if language is None:
        language = settings.es_analyzer
    mapping = _build_es_mapping(embedding_dim, language)

    return ElasticsearchDocumentStore(
        hosts=f"{scheme}://{host}:{port}",
        custom_mapping=mapping,
        index=index,
    )


def _build_es_mapping(embedding_dim: int, language: str) -> Dict[str, Any]:
    return {
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
            "content": {"type": "text", "analyzer": language},
            "embedding": {
                "type": "dense_vector",
                "dims": embedding_dim,
                "index": True,
                "similarity": "cosine",
            },
            "blob": {"type": "binary"},
            "score": {"type": "float"},
        },
    }


def configure_session(
    *,
    es_host: str,
    es_port: int,
    es_scheme: str = "http",
    es_index: str = "documents",
    s3_endpoint_url: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_use_ssl: bool = False,
    embedding_dim: Optional[int] = None,
) -> None:
    """Replace the module-level session with freshly constructed instances.

    Called by `needle.connect()` when the user supplies explicit connection
    parameters.  Also useful for testing: inject mocks by assigning directly
    to `_session.document_store` / `_session.s3_storage`.
    """
    _session.document_store = _build_document_store(
        host=es_host,
        port=es_port,
        scheme=es_scheme,
        index=es_index,
        embedding_dim=embedding_dim,
    )
    _session.s3_storage = S3Storage(
        endpoint_url=s3_endpoint_url,
        access_key=s3_access_key,
        secret_key=s3_secret_key,
        use_ssl=s3_use_ssl,
    )


class S3Storage:
    """S3-compatible storage class with basic file operations."""

    def __init__(
        self,
        endpoint_url: str = None,
        access_key: str = None,
        secret_key: str = None,
        use_ssl: bool = None,
    ):
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            use_ssl=use_ssl,
        )

    def _create_bucket_if_not_exists(self, bucket: str):
        try:
            self.client.head_bucket(Bucket=bucket)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                # The bucket does not exist, so create it
                self.client.create_bucket(Bucket=bucket)
            else:
                raise

    def upload_file(
        self,
        bucket: str,
        obj: str,
        data: bytes,
        content_type: str,
        metadata: Dict[str, str] = dict(),
    ):
        """Upload a file to an S3 bucket."""
        self._create_bucket_if_not_exists(bucket)
        self.client.put_object(
            Bucket=bucket,
            Key=obj,
            Body=data,
            ContentType=content_type,
            Metadata=metadata,
        )

    def list_files(self, bucket: str) -> List[Dict[str, Any]]:
        """List all files in an S3 bucket."""
        self._create_bucket_if_not_exists(bucket)
        response = self.client.list_objects_v2(Bucket=bucket)

        # Native metadata retrieval isn't supported by list_objects_v2,
        # so we need to call head_object for each file
        files = []
        for item in response.get("Contents", []):
            key = item["Key"]
            head = self.client.head_object(Bucket=bucket, Key=key)
            files.append(
                {
                    "key": key,
                    "size": item["Size"],
                    "last_modified": item["LastModified"],
                    "content_type": head["ContentType"],
                    "metadata": head.get("Metadata", {}),
                }
            )

        return files

    def delete_file(self, bucket: str, obj: str):
        """Delete a file from an S3 bucket."""
        try:
            self.client.head_object(Bucket=bucket, Key=obj)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise FileNotFoundError(f"File '{obj}' does not exist in bucket '{bucket}'.")
            else:
                raise

        self.client.delete_object(Bucket=bucket, Key=obj)

    def get_file(self, bucket: str, obj: str) -> bytes:
        """Get a file from an S3 bucket."""
        try:
            response = self.client.get_object(Bucket=bucket, Key=obj)
            return response["Body"].read()
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                raise FileNotFoundError(f"File '{obj}' does not exist in bucket '{bucket}'.")
            else:
                raise


# Backwards-compatible module-level mapping constant used by test fixtures and
# any external code that imported ES_MAPPING directly.
# TODO: Migrate to newer method of getting the mapping via get_document_store() and remove this constant.
ES_MAPPING = _build_es_mapping(settings.embedding_dim, settings.es_analyzer)


def get_document_store() -> ElasticsearchDocumentStore:
    """Return the active ElasticsearchDocumentStore from the session."""
    return _session.document_store


def get_s3_storage() -> S3Storage:
    """Return the active S3Storage from the session."""
    return _session.s3_storage


# TODO: Implement timeout and retries
def healthcheck(timeout: int = 5) -> Dict[str, bool]:
    """Check connectivity to the active session's services (ES and S3).

    Returns a dict with boolean statuses, does NOT raise on failure.
    """
    statuses: Dict[str, bool] = {"elasticsearch": False, "s3": False}

    try:
        es_client = _session.document_store.client  # type: ignore[attr-defined]
        if hasattr(es_client, "ping") and es_client.ping():
            statuses["elasticsearch"] = True
    except Exception:
        statuses["elasticsearch"] = False

    try:
        _session.s3_storage.client.list_buckets()
        statuses["s3"] = True
    except Exception:
        statuses["s3"] = False

    return statuses
