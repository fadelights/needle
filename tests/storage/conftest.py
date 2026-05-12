"""Shared fixtures for testing."""

import os

import pytest
from moto import mock_aws

from needle.storage import S3Storage


@pytest.fixture(scope="session")
def aws_credentials():
    """Mocked AWS credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "test_user"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test_password"


@pytest.fixture(scope="function")
def s3_storage(aws_credentials):
    with mock_aws():
        # We don't need a real endpoint_url here because moto
        # intercepts the standard boto3.client("s3") call.
        storage = S3Storage(
            access_key=os.environ["AWS_ACCESS_KEY_ID"],
            secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )

        yield storage


@pytest.fixture
def test_bucket():
    obj = "oliver-twist.txt"
    content = b"""
    Among other public buildings in a certain town, which for many reasons
    it will be prudent to refrain from mentioning, and to which I will
    assign no fictitious name, there is one anciently common to most towns,
    great or small: to wit, a workhouse; and in this workhouse was born; on
    a day and date which I need not trouble myself to repeat, inasmuch as
    it can be of no possible consequence to the reader, in this stage of
    the business at all events; the item of mortality whose name is
    prefixed to the head of this chapter.
    """
    bucket = "11111111-2222-3333-4444-555555555555"

    return obj, content, bucket
