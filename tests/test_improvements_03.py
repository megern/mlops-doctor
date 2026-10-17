from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def payload(a, b, kind="number", enabled=True):
    return {
        "train": [{"id": f"a{i}", "x": x, "y": 0} for i, x in enumerate(a)],
        "validation": [{"id": f"b{i}", "x": x, "y": 0} for i, x in enumerate(b)],
        "schema": {"id": "string", "x": kind, "y": "number"},
        "id_column": "id",
        "target": "y",
        "features": ["x"],
        "distribution_checks": enabled,
        "distribution_threshold": 0.25,
    }


def test_equal_mean_shape_shift_is_detected_only_when_opted_in():
    data = payload([-1, 1], [-2, 2])
    r = client.post("/analyze", json=data).json()
    assert r["numeric_drift"][0]["standardized_mean_shift"] == 0
    assert r["distribution_drift"][0]["distance"] == 0.5
    assert {f["code"] for f in r["findings"]} == {"DISTRIBUTION_DRIFT"}
    del data["distribution_checks"]
    r = client.post("/analyze", json=data).json()
    assert r["distribution_drift"] == [] and r["healthy"]


def test_empirical_cdf_ties_and_unequal_sample_sizes():
    result = client.post("/analyze", json=payload([0, 0, 1, 1], [0, 1])).json()
    assert result["distribution_drift"][0]["distance"] == 0
    result = client.post("/analyze", json=payload([0, 0], [1, 1, 1])).json()
    assert result["distribution_drift"][0]["distance"] == 1


def test_categorical_total_variation_and_boolean_categories():
    result = client.post(
        "/analyze", json=payload(["a", "a", "b", "b"], ["a", "a", "a", "b"], "string")
    ).json()
    assert result["distribution_drift"][0]["distance"] == 0.25
    assert not result["distribution_drift"][0]["flagged"]
    result = client.post(
        "/analyze", json=payload([True, False], [False, False], "boolean")
    ).json()
    assert result["distribution_drift"][0]["distance"] == 0.5
    assert result["distribution_drift"][0]["metric"] == "total_variation"


def test_missing_and_invalid_values_skip_distribution_not_imputed():
    result = client.post("/analyze", json=payload([1, None], [1, 2])).json()
    assert result["distribution_drift"][0]["distance"] is None
    assert result["distribution_drift"][0]["status"] == "skipped_invalid_values"
    assert any(f["code"] == "SCHEMA_MISMATCH" for f in result["findings"])
    for threshold in [-0.1, 1.1]:
        data = payload([0], [1])
        data["distribution_threshold"] = threshold
        assert client.post("/analyze", json=data).status_code == 422


def test_csv_opt_in_reaches_same_distribution_computation():
    data = {
        "train_csv": "id,x,y\na,-1,0\nb,1,0",
        "validation_csv": "id,x,y\nc,-2,0\nd,2,0",
        "schema": {"id": "string", "x": "number", "y": "number"},
        "id_column": "id",
        "target": "y",
        "features": ["x"],
        "distribution_checks": True,
        "distribution_threshold": 0,
    }
    r = client.post("/csv/analyze", json=data)
    assert r.status_code == 200
    assert r.json()["distribution_drift"][0]["distance"] == 0.5
