from fastapi.testclient import TestClient
from app.main import app
from app.core import analyze, Request

client = TestClient(app)


def payload():
    return {
        "train": [{"id": "a", "x": 1, "y": 0}, {"id": "b", "x": 3, "y": 1}],
        "validation": [{"id": "c", "x": 2, "y": 0}],
        "schema": {"id": "string", "x": "number", "y": "number"},
        "id_column": "id",
        "target": "y",
        "features": ["x"],
    }


def test_healthy_known_mean():
    result = analyze(Request(**payload()))
    assert result["healthy"]
    assert result["numeric_drift"][0]["standardized_mean_shift"] == 0


def test_leakage_schema_and_drift():
    data = payload()
    data["features"] = ["x", "y"]
    data["validation"] = [{"id": "a", "x": 10, "y": False}]
    result = client.post("/analyze", json=data)
    assert result.status_code == 200
    codes = {f["code"] for f in result.json()["findings"]}
    assert {"SPLIT_LEAKAGE", "TARGET_LEAKAGE", "SCHEMA_MISMATCH", "MEAN_DRIFT"} <= codes


def test_constant_baseline_changed():
    data = payload()
    data["train"][1]["x"] = 1
    r = client.post("/analyze", json=data).json()
    assert r["numeric_drift"][0]["constant_reference_changed"]
    assert r["numeric_drift"][0]["standardized_mean_shift"] is None


def test_reject_bad_manifest_and_empty_split():
    data = payload()
    data["features"] = ["absent"]
    assert client.post("/analyze", json=data).status_code == 422
    data = payload()
    data["train"] = []
    assert client.post("/analyze", json=data).status_code == 422
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_ids_preserve_type_and_duplicates_are_reported():
    data = payload()
    data["schema"]["id"] = "number"
    data["train"][0]["id"] = 1
    data["train"][1]["id"] = 1
    data["validation"][0]["id"] = "1"
    codes = {f["code"] for f in client.post("/analyze", json=data).json()["findings"]}
    assert "DUPLICATE_ID" in codes
    assert "SPLIT_LEAKAGE" not in codes
    assert "SCHEMA_MISMATCH" in codes
