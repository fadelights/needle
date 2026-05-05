from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError
from haystack_integrations.document_stores.elasticsearch import (
    ElasticsearchDocumentStore,
)

from .config import settings


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
        metadata: dict[str, str] = None,
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
                raise FileNotFoundError(
                    f"File '{obj}' does not exist in bucket '{bucket}'."
                )
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
                raise FileNotFoundError(
                    f"File '{obj}' does not exist in bucket '{bucket}'."
                )
            else:
                raise


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

s3_storage = S3Storage(
    endpoint_url=f"http://{settings.minio_host}:{settings.minio_port}",
    access_key=settings.minio_root_user,
    secret_key=settings.minio_root_password,
    use_ssl=settings.minio_secure,
)
