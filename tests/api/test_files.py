import io
import uuid


def test_list_files(client):
    client, mock_storage, *_ = client

    mock_storage.list_files.return_value = [
        {
            "key": "7a6d9d27-c9c7-4c6b-b187-d82971d20890.pdf",
            "size": 9304353,
            "last_modified": "2026-05-10T08:11:26.260000Z",
            "content_type": "application/pdf",
            "metadata": {"original-path": "A Tale of Two Cities.pdf"},
        }
    ]

    response = client.get("/api/files")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["metadata"]["original-path"] == "A Tale of Two Cities.pdf"
    mock_storage.list_files.assert_called_once()


def test_upload_supported_file(client):
    client, mock_storage, mock_indexer, _ = client

    file_content = b"Hello, PyTest!"
    file_name = "test.txt"

    response = client.post(
        "/api/files",
        files={"file": (file_name, io.BytesIO(file_content), "text/plain")},
    )

    assert response.status_code == 201
    assert response.json()["message"] == "Document uploaded successfully."

    mock_storage.upload_file.assert_called_once()
    mock_indexer.run.assert_called_once()


def test_upload_unsupported_file(client):
    client, *_ = client

    response = client.post(
        "/api/files",
        files={
            "file": (
                "test.exe",
                b"binary data",
                "application/vnd.microsoft.portable-executable",
            )
        },
    )

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_get_file_content(client):
    client, mock_storage, *_ = client

    mock_content = "dummy file content"
    mock_storage.get_file.return_value = mock_content.encode("utf-8")
    storage_path = str(uuid.uuid4())

    response = client.get(f"/api/files/{storage_path}")

    assert response.status_code == 200
    assert response.json()["content"] == mock_content
    assert response.json()["storage_path"] == storage_path


def test_get_file_content_not_found(client):
    client, mock_storage, *_ = client

    mock_storage.get_file.side_effect = FileNotFoundError()

    response = client.get("/api/files/missing.txt")

    assert response.status_code == 404


def test_update_file_content(client):
    client, mock_storage, mock_indexer, _ = client

    storage_path = f"{uuid.uuid4()}.txt"
    new_content = "very shiny!"
    mock_storage.list_files.return_value = [
        {"key": storage_path, "metadata": {"original-path": "test.txt"}}
    ]

    response = client.put(f"/api/files/{storage_path}", json={"content": new_content})

    assert response.status_code == 200
    assert "updated and re-indexed" in response.json()["message"]
    mock_storage.delete_file.assert_called_once()
    mock_storage.upload_file.assert_called_once()
    mock_indexer.run.assert_called_once()


def test_update_file_content_unsupported_file(client):
    client, *_ = client

    response = client.put(
        "/api/files/book.pdf", json={"content": "What's up teammates!"}
    )

    assert response.status_code == 400


def test_update_file_not_found(client):
    client, mock_storage, *_ = client

    mock_storage.list_files.return_value = []

    response = client.put(
        "/api/files/rules.md", json={"content": "There are no rules!"}
    )

    assert response.status_code == 404


def test_delete_file(client):
    client, mock_storage, *_ = client

    response = client.delete("/api/files/test.pdf")

    assert response.status_code == 200
    assert "success" in response.json()["message"]
    mock_storage.delete_file.assert_called_once()


def test_delete_file_not_found(client):
    client, mock_storage, *_ = client

    mock_storage.delete_file.side_effect = FileNotFoundError()
    response = client.delete("/api/files/test.pdf")

    assert response.status_code == 200  # the action should still succeed
    assert "does not exist" in response.json()["message"]
    mock_storage.delete_file.assert_called_once()
