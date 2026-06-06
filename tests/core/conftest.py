"""Shared fixtures for core tests."""

from unittest.mock import Mock, patch

import pytest


@pytest.fixture(scope="function")
def core_mocks():
    """Patch external dependencies and yield their mocks."""
    mock_storage = Mock()
    mock_document_store = Mock()
    mock_indexing_pipeline = Mock()
    mock_query_pipeline = Mock()

    with (
        patch("needle.core.get_s3_storage", return_value=mock_storage),
        patch("needle.core.get_document_store", return_value=mock_document_store),
        patch("needle.core.get_indexing_pipeline", return_value=mock_indexing_pipeline),
        patch("needle.core.get_query_pipeline", return_value=mock_query_pipeline),
    ):
        yield mock_storage, mock_document_store, mock_indexing_pipeline, mock_query_pipeline
