"""Shared fixtures for testing document and object stores."""

import os

import pytest
from haystack_integrations.document_stores.elasticsearch import (
    ElasticsearchDocumentStore,
)
from moto import mock_aws

from needle.config import settings
from needle.storage import S3Storage


# ---------------------------------------------------------------------------
# Elasticsearch document store
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def es_document_store():
    """
    Real ElasticsearchDocumentStore pointed at the configured ES instance.

    Expects ES to be reachable at the configured ENV host. If not, will
    use whatever defaults set for the app. It will use a dummy index
    for performing the tests.

    The index is dropped after the session so tests stay idempotent.
    """
    index = "test"
    from needle.storage import ES_MAPPING

    store = ElasticsearchDocumentStore(
        hosts=f"{settings.es_scheme}://{settings.es_host}:{settings.es_port}",
        custom_mapping=ES_MAPPING,
        index=index,
    )

    yield store

    # Teardown
    try:
        if store._client is not None:
            store._client.indices.delete(index=index, ignore_unavailable=True)
    except Exception as exc:
        raise Exception(f"ES teardown failed due to the following reason: {exc}")


# ---------------------------------------------------------------------------
# S3 object storage
# ---------------------------------------------------------------------------
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
