from fastapi.testclient import TestClient
from app.main import app
from app.core import empirical_cdf_distance, total_variation


def payload():
    return {
        "train": [{"id": i, "x": x, "y": 0} for i, x in enumerate([-2, -2, 2, 2])],
        "validation": [{"id": i + 4, "x": 0, "y": 0} for i in range(4)],
        "schema": {"id": "number", "x": "number", "y": "number"},
        "id_column": "id",
        "target": "y",
        "features": ["x"],
        "distribution_checks": True,
        "distribution_threshold": 0.5,
    }


def test_independent_distribution_distances_and_strict_boundary():
    assert empirical_cdf_distance([-2, -2, 2, 2], [0, 0, 0, 0]) == 0.5
    assert empirical_cdf_distance([1, 1, 2], [1, 2, 2]) == 1 / 3
    assert total_variation(["A", "A", "B", "B"], ["A", "A", "A", "B"]) == 0.25
    client = TestClient(app)
    data = payload()
    result = client.post("/analyze", json=data).json()
    assert result["numeric_drift"][0]["standardized_mean_shift"] == 0
    assert result["distribution_drift"][0]["distance"] == 0.5
    assert not result["distribution_drift"][0]["flagged"]
    data["distribution_threshold"] = 0.49
    result = client.post("/analyze", json=data).json()
    assert result["distribution_drift"][0]["flagged"]
    assert any(f["code"] == "DISTRIBUTION_DRIFT" for f in result["findings"])


def test_independent_invalid_cells_are_explicitly_skipped_without_imputation():
    data = payload()
    data["validation"][0]["x"] = None
    response = TestClient(app).post("/analyze", json=data)
    assert response.status_code == 200
    result = response.json()
    distribution = result["distribution_drift"][0]
    assert (
        distribution["distance"] is None
        and distribution["status"] == "skipped_invalid_values"
    )
    assert distribution["train_count"] == 4 and distribution["validation_count"] == 4
    assert any(f["code"] == "SCHEMA_MISMATCH" for f in result["findings"])
