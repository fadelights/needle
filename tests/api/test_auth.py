def test_register_new_business(client):
    client, *_ = client
    response = client.post(
        "/api/auth/register",
        json={"name": "randomcorp", "password": "secret"},
    )

    assert response.status_code == 201
    assert response.json()["name"] == "randomcorp"
    assert len(response.json()["business_id"]) == 36  # UUID4 length


def test_register_duplicate_business(client):
    client, *_ = client
    response = client.post(
        "/api/auth/register",
        json={"name": "acme", "password": "secret"},  # already created in fixtures
    )

    assert response.status_code == 400


def test_login_correct_password(client):
    client, *_ = client
    response = client.post(
        "/api/auth/token",
        data={"username": "acme", "password": "secret"},
    )

    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"


def test_login_incorrect_password(client):
    client, *_ = client
    response = client.post(
        "/api/auth/token",
        data={"username": "acme", "password": "idk"},
    )

    assert response.status_code == 401
