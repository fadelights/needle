from unittest.mock import MagicMock, call, patch

import pytest

import needle
import needle.storage as storage_module
from needle.config import settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def pipeline_cooldown():
    """Prevent real pipeline construction."""
    with (
        patch("needle.IndexingPipeline"),
        patch("needle.QueryPipeline"),
    ):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConnectDefaultParams:
    """`connect()` with no arguments uses settings values."""

    def test_healthcheck(self):
        mock_healthcheck = MagicMock(return_value={"elasticsearch": True, "s3": True})
        with (
            patch("needle.configure_session"),
            patch("needle.healthcheck", mock_healthcheck),
        ):
            result = needle.connect()

        assert result == {"elasticsearch": True, "s3": True}
        mock_healthcheck.assert_called_once_with(timeout=5)

    def test_healthcheck_timeout(self):
        mock_healthcheck = MagicMock(return_value={})
        with (
            patch("needle.configure_session"),
            patch("needle.healthcheck", mock_healthcheck),
        ):
            needle.connect(timeout=10)

        mock_healthcheck.assert_called_once_with(timeout=10)

    def test_configure_session(self):
        mock_configure = MagicMock()
        with (
            patch("needle.configure_session", mock_configure),
            patch("needle.healthcheck", return_value={}),
        ):
            needle.connect()

        mock_configure.assert_called_once_with(
            es_host=settings.es_host,
            es_port=settings.es_port,
            es_scheme=settings.es_scheme,
            es_index=settings.es_index,
            s3_endpoint_url=f"{'https' if settings.minio_secure else 'http'}://{settings.minio_host}:{settings.minio_port}",
            s3_access_key=settings.minio_root_user,
            s3_secret_key=settings.minio_root_password,
            s3_use_ssl=settings.minio_secure,
        )


class TestConnectExplicitParams:
    """`connect()` with explicit params overrides defaults."""

    def test_es_params(self):
        mock_configure = MagicMock()
        with (
            patch("needle.configure_session", mock_configure),
            patch("needle.healthcheck", return_value={}),
        ):
            needle.connect(es_host="noodle-es", es_port=9201, es_scheme="https", es_index="test")

        kwargs = mock_configure.call_args.kwargs
        assert kwargs["es_host"] == "noodle-es"
        assert kwargs["es_port"] == 9201
        assert kwargs["es_scheme"] == "https"
        assert kwargs["es_index"] == "test"

    def test_s3_params(self):
        mock_configure = MagicMock()
        with (
            patch("needle.configure_session", mock_configure),
            patch("needle.healthcheck", return_value={}),
        ):
            needle.connect(
                s3_endpoint_url="http://noodle-s3:9000",
                s3_access_key="someuser",
                s3_secret_key="somesecret",
                s3_use_ssl=True,
            )

        kwargs = mock_configure.call_args.kwargs
        assert kwargs["s3_endpoint_url"] == "http://noodle-s3:9000"
        assert kwargs["s3_access_key"] == "someuser"
        assert kwargs["s3_secret_key"] == "somesecret"
        assert kwargs["s3_use_ssl"] is True

    def test_explicit_params(self):
        """Explicit params should not be overwritten by default settings."""
        mock_configure = MagicMock()
        custom_url = "http://noodle-endpoint:1234"
        with (
            patch("needle.configure_session", mock_configure),
            patch("needle.healthcheck", return_value={}),
        ):
            needle.connect(s3_endpoint_url=custom_url)

        assert mock_configure.call_args.kwargs["s3_endpoint_url"] == custom_url

    def test_mixed_params(self):
        """Only the supplied params are overridden; the rest stay as settings."""
        mock_configure = MagicMock()
        with (
            patch("needle.configure_session", mock_configure),
            patch("needle.healthcheck", return_value={}),
        ):
            needle.connect(es_host="override-host")

        kwargs = mock_configure.call_args.kwargs
        assert kwargs["es_host"] == "override-host"
        # everything else falls back to settings
        assert kwargs["es_port"] == settings.es_port
        assert kwargs["es_index"] == settings.es_index
        assert kwargs["s3_access_key"] == settings.minio_root_user


class TestConnectSessionReplaced:
    """`connect()` actually replaces the session objects inside storage."""

    def test_configure_session_replaces_storages(self):
        """After `connect()`, `get_document_store()` returns the new instance."""
        fake_ds = MagicMock()
        fake_s3 = MagicMock()

        def fake_configure_session(**kwargs):
            storage_module._session.document_store = fake_ds
            storage_module._session.s3_storage = fake_s3

        with (
            patch("needle.configure_session", side_effect=fake_configure_session),
            patch("needle.healthcheck", return_value={"elasticsearch": True, "s3": True}),
        ):
            needle.connect()

        assert storage_module.get_document_store() is fake_ds
        assert storage_module.get_s3_storage() is fake_s3

    def test_healthcheck_called_after_configure_session(self):
        """`healthcheck()` must be called *after* `configure_session()`."""
        call_order = []

        def fake_configure_session(**kwargs):
            call_order.append("configure")

        def fake_healthcheck(**kwargs):
            call_order.append("healthcheck")
            return {}

        with (
            patch("needle.configure_session", side_effect=fake_configure_session),
            patch("needle.healthcheck", side_effect=fake_healthcheck),
        ):
            needle.connect()

        assert call_order == ["configure", "healthcheck"]


class TestConfigureSession:
    """Unit tests for storage.configure_session() in isolation."""

    def test_storage_setting(self):
        fake_ds = MagicMock()
        fake_s3 = MagicMock()

        with (
            patch("needle.storage._build_document_store", return_value=fake_ds),
            patch("needle.storage.S3Storage", return_value=fake_s3),
        ):
            storage_module.configure_session(
                es_host="localhost",
                es_port=9200,
                es_scheme="http",
                es_index="test",
                s3_endpoint_url="http://localhost:9000",
                s3_access_key="user",
                s3_secret_key="pass",
                s3_use_ssl=False,
            )

        assert storage_module._session.document_store is fake_ds
        assert storage_module._session.s3_storage is fake_s3


class TestConnectWarmup:
    """`connect(warmup=...)` controls pipeline construction and session caching."""

    def test_warmup_true(self):
        """Warmed-up pipeline instances must be placed on the session."""
        fake_indexing = MagicMock()
        fake_query = MagicMock()

        with (
            patch("needle.configure_session"),
            patch("needle.healthcheck", return_value={}),
            patch("needle.IndexingPipeline", return_value=fake_indexing) as mock_index_cls,
            patch("needle.QueryPipeline", return_value=fake_query) as mock_query_cls,
        ):
            needle.connect(warmup=True)

        mock_index_cls.assert_called_once_with(warmup=True)
        mock_query_cls.assert_called_once_with(warmup=True)
        assert storage_module._session.indexing_pipeline is fake_indexing
        assert storage_module._session.query_pipeline is fake_query

    def test_warmup_false(self):
        """`connect(warmup=False)` must not instantiate any pipeline."""
        with (
            patch("needle.configure_session"),
            patch("needle.healthcheck", return_value={}),
            patch("needle.IndexingPipeline") as mock_index_cls,
            patch("needle.QueryPipeline") as mock_query_cls,
        ):
            needle.connect(warmup=False)

        mock_index_cls.assert_not_called()
        mock_query_cls.assert_not_called()

    def test_warmup_true_is_default(self):
        """Calling `connect()` without `warmup=` should behave as `warmup=True`."""
        with (
            patch("needle.configure_session"),
            patch("needle.healthcheck", return_value={}),
            patch("needle.IndexingPipeline") as mock_index_cls,
            patch("needle.QueryPipeline") as mock_query_cls,
        ):
            needle.connect()

        mock_index_cls.assert_called_once_with(warmup=True)
        mock_query_cls.assert_called_once_with(warmup=True)

    def test_cached_pipeline_reused_by_get_indexing_pipeline(self):
        """`get_indexing_pipeline()` returns the cached pipeline from the session."""
        from needle.pipelines import get_indexing_pipeline

        fake_pipeline = MagicMock()
        storage_module._session.indexing_pipeline = fake_pipeline

        result = get_indexing_pipeline()

        assert result is fake_pipeline

    def test_cached_pipeline_reused_by_get_query_pipeline(self):
        """`get_query_pipeline()` returns the cached pipeline from the session."""
        from needle.pipelines import get_query_pipeline

        fake_pipeline = MagicMock()
        storage_module._session.query_pipeline = fake_pipeline

        result = get_query_pipeline()

        assert result is fake_pipeline

    def test_warmup_true_replaces_cached_pipelines(self):
        """A second `connect(warmup=True)` must replace the previously cached pipelines."""
        old_indexing = MagicMock()
        old_query = MagicMock()
        storage_module._session.indexing_pipeline = old_indexing
        storage_module._session.query_pipeline = old_query

        new_indexing = MagicMock()
        new_query = MagicMock()

        with (
            patch("needle.configure_session"),
            patch("needle.healthcheck", return_value={}),
            patch("needle.IndexingPipeline", return_value=new_indexing),
            patch("needle.QueryPipeline", return_value=new_query),
        ):
            needle.connect(warmup=True)

        assert storage_module._session.indexing_pipeline is new_indexing
        assert storage_module._session.query_pipeline is new_query

    def test_warmup_false_clears_cached_pipelines(self):
        """Reconnecting with `warmup=False` should evict stale cached pipelines."""
        storage_module._session.indexing_pipeline = MagicMock()
        storage_module._session.query_pipeline = MagicMock()

        with (
            patch("needle.configure_session"),
            patch("needle.healthcheck", return_value={}),
            patch("needle.IndexingPipeline"),
            patch("needle.QueryPipeline"),
        ):
            needle.connect(warmup=False)

        assert storage_module._session.indexing_pipeline is None
        assert storage_module._session.query_pipeline is None
