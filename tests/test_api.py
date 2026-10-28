from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 35,
    "sex": "female",
    "bmi": 27.5,
    "children": 2,
    "smoker": "no",
    "region": "northeast",
}


def test_root_returns_message():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_input():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_charges" in body
    assert isinstance(body["predicted_charges"], float)
    assert body["predicted_charges"] > 0


def test_predict_missing_field_returns_422():
    payload = dict(VALID_PAYLOAD)
    del payload["smoker"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "smoker"


def test_predict_age_out_of_range_returns_422():
    payload = dict(VALID_PAYLOAD, age=10)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "age"


def test_predict_invalid_category_returns_422():
    payload = dict(VALID_PAYLOAD, smoker="maybe")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_wrong_type_returns_422():
    payload = dict(VALID_PAYLOAD, age="abc")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_get_not_allowed_returns_405():
    response = client.get("/predict")
    assert response.status_code == 405


def test_unknown_route_returns_404():
    response = client.get("/does-not-exist")
    assert response.status_code == 404