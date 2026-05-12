import pytest
from botocore.exceptions import ClientError


def test_upload_and_list_files(s3_storage, test_bucket):
    obj, content, bucket = test_bucket

    s3_storage.upload_file(
        bucket=bucket,
        obj=obj,
        data=content,
        content_type="text/plain",
        metadata={"Original-Path": obj},
    )

    files = s3_storage.list_files(bucket=bucket)

    assert len(files) == 1
    assert files[0]["key"] == obj
    assert files[0]["metadata"] == {"original-path": obj}


def test_get_file(s3_storage, test_bucket):
    obj, content, bucket = test_bucket

    s3_storage.upload_file(
        bucket=bucket,
        obj=obj,
        data=content,
        content_type="text/plain",
        metadata={"Original-Path": obj},
    )

    content_bytes = s3_storage.get_file(bucket=bucket, obj=obj)

    assert content_bytes == content


def test_get_file_not_found(s3_storage, test_bucket):
    *_, bucket = test_bucket

    s3_storage._create_bucket_if_not_exists(bucket)
    with pytest.raises(FileNotFoundError) as excinfo:
        s3_storage.get_file(bucket, "nonexistent.md")

    assert "does not exist" in str(excinfo.value)


def test_get_file_bucket_not_found(s3_storage):
    with pytest.raises(ClientError) as excinfo:
        s3_storage.get_file("nonexistent", "nonexistent.md")

    assert excinfo.value.response["Error"]["Code"] == "NoSuchBucket"


def test_delete_file(s3_storage, test_bucket):
    *_, bucket = test_bucket

    obj = "deleteme.docx"
    s3_storage.upload_file(bucket, obj, b"...", "text/plain")

    assert len(s3_storage.list_files(bucket)) == 1

    s3_storage.delete_file(bucket, obj)

    assert len(s3_storage.list_files(bucket)) == 0


def test_delete_file_not_found(s3_storage, test_bucket):
    *_, bucket = test_bucket

    s3_storage._create_bucket_if_not_exists(bucket)
    with pytest.raises(FileNotFoundError) as excinfo:
        s3_storage.delete_file(bucket=bucket, obj="deleteme.xlsx")

    assert "does not exist" in str(excinfo.value)
