"""Shared fixtures for testing."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import app
from balequeue.auth import get_current_business, get_db, get_passwd_hash
from balequeue.database import Base
from balequeue.models import Business

SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@pytest.fixture(scope="function")
def db_session():
    """Create a new database session for testing."""
    Base.metadata.create_all(bind=engine)
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def test_business(db_session):
    """Pre-created business for use in tests."""
    business = Business(name="acme", hashed_password=get_passwd_hash("secret"))
    db_session.add(business)
    db_session.commit()
    db_session.refresh(business)

    return business


@pytest.fixture(scope="function")
def client(db_session, test_business):
    """Test client with mocked dependencies."""

    def override_get_db():
        yield db_session

    def override_get_current_business():
        return test_business

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_business] = override_get_current_business

    # Patch storage and pipelines so tests never touch MinIO or Elasticsearch
    with (
        patch("balequeue.api.s3_storage") as mock_storage,
        patch("balequeue.api.indexing_pipeline") as mock_indexer,
        patch("balequeue.api.query_pipeline") as mock_query,
    ):
        yield TestClient(app), mock_storage, mock_indexer, mock_query

    app.dependency_overrides.clear()
