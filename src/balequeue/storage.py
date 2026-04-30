import boto3
from botocore.exceptions import ClientError
from haystack_integrations.document_stores.elasticsearch import (
    ElasticsearchDocumentStore,
)

from .config import settings


class S3Storage:
    """S3-compatible storage class for MinIO."""

    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=f"http://{settings.minio_host}:{settings.minio_port}",
            aws_access_key_id=settings.minio_root_user,
            aws_secret_access_key=settings.minio_root_password,
            use_ssl=settings.minio_secure,
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

    def upload_file(self, bucket: str, obj: str, data: bytes, content_type: str):
        """Upload a file to an S3 bucket."""
        self._create_bucket_if_not_exists(bucket)
        self.client.put_object(
            Bucket=bucket,
            Key=obj,
            Body=data,
            ContentType=content_type,
        )


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

s3_storage = S3Storage()
