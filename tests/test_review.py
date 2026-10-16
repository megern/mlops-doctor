from fastapi.testclient import TestClient
from app.main import app


def payload():
    return {
        "train": [{"id": "a", "x": 0, "y": 0}, {"id": "b", "x": 2, "y": 1}],
        "validation": [{"id": "c", "x": 4, "y": 0}],
        "schema": {"id": "string", "x": "number", "y": "number"},
        "id_column": "id",
        "target": "y",
        "features": ["x"],
    }


def test_review_known_mean_shift():
    response = TestClient(app).post("/analyze", json=payload())
    assert response.status_code == 200
    assert response.json()["numeric_drift"][0]["standardized_mean_shift"] == 3
    assert any(f["code"] == "MEAN_DRIFT" for f in response.json()["findings"])


def test_review_explicit_leakage():
    data = payload()
    data["validation"][0]["id"] = "a"
    data["features"] = ["y"]
    result = TestClient(app).post("/analyze", json=data).json()
    codes = {f["code"] for f in result["findings"]}
    assert {"TARGET_LEAKAGE", "SPLIT_LEAKAGE"} <= codes


def test_review_overflow_effect_is_controlled():
    data = payload()
    data["train"][0]["x"] = 1e-300
    data["train"][1]["x"] = 2e-300
    data["validation"][0]["x"] = 1e100
    response = TestClient(app, raise_server_exceptions=False).post(
        "/analyze", json=data
    )
    assert response.status_code in (200, 422)
    if response.status_code == 200:
        assert response.json()["numeric_drift"][0]["standardized_mean_shift"] is None


def test_review_huge_integer_controlled():
    data = payload()
    data["train"][0]["x"] = 10**400
    response = TestClient(app, raise_server_exceptions=False).post(
        "/analyze", json=data
    )
    assert response.status_code in (200, 422)
    if response.status_code == 200:
        assert any(f["code"] == "SCHEMA_MISMATCH" for f in response.json()["findings"])
