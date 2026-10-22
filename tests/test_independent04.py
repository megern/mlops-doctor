import math
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_known_classification_matrix_macro_weighted_and_slice_denominators():
    payload = {
        "task": "classification",
        "actual": ["a", "a", "b", "b"],
        "predicted": ["a", "b", "b", "c"],
        "groups": ["large", "large", "tiny-a", "tiny-b"],
        "minimum_support": 2,
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    result = response.json()
    overall = result["overall"]
    assert overall["confusion_matrix"] == [[1, 1, 0], [0, 1, 1], [0, 0, 0]]
    assert overall["accuracy"] == 0.5
    assert overall["macro"]["f1"] == pytest.approx(7 / 18)
    assert overall["weighted"]["f1"] == pytest.approx(7 / 12)
    large = next(row for row in result["slices"] if row["group"] == "large")
    assert large["metrics"]["macro"]["f1"] == pytest.approx(2 / 9)
    assert all(
        row["metrics"] is None for row in result["slices"] if row["group"] != "large"
    )


def test_known_regression_errors_and_constant_slice_reason():
    payload = {
        "task": "regression",
        "actual": [1, 2, 3],
        "predicted": [1, 2, 5],
        "groups": ["a", "a", "b"],
        "minimum_support": 1,
    }
    result = client.post("/evaluate", json=payload).json()
    assert result["overall"]["mae"] == pytest.approx(2 / 3)
    assert result["overall"]["rmse"] == pytest.approx(math.sqrt(4 / 3))
    assert result["overall"]["r2"] == -1
    constant = next(row for row in result["slices"] if row["group"] == "b")["metrics"]
    assert (
        constant["r2"] is None
        and constant["r2_undefined_reason"] == "constant actual values"
    )


def test_adjacent_large_integer_labels_and_prediction_errors_are_not_rounded_away():
    left, right = 9007199254740992, 9007199254740993
    response = client.post(
        "/evaluate",
        json={"task": "classification", "actual": [left], "predicted": [right]},
    )
    assert response.status_code == 200
    result = response.json()["overall"]
    assert result["accuracy"] == 0 and len(result["labels"]) == 2
    result = client.post(
        "/evaluate", json={"task": "regression", "actual": [left], "predicted": [right]}
    ).json()["overall"]
    assert result["mae"] == 1 and result["rmse"] == 1


def test_subnormal_regression_variance_uses_stable_centering():
    unit = float.fromhex("0x0.0000000000001p-1022")
    result = client.post(
        "/evaluate",
        json={"task": "regression", "actual": [0, unit], "predicted": [unit, 0]},
    ).json()["overall"]
    assert result["r2"] == pytest.approx(-3)


def test_invalid_lengths_and_unbounded_label_group_counts_are_rejected():
    assert (
        client.post(
            "/evaluate", json={"task": "regression", "actual": [1], "predicted": [1, 2]}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/evaluate", json={"task": "regression", "actual": [True], "predicted": [1]}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/evaluate",
            json={
                "task": "classification",
                "actual": list(range(51)),
                "predicted": list(range(51)),
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/evaluate",
            json={
                "task": "regression",
                "actual": [1] * 101,
                "predicted": [1] * 101,
                "groups": [str(i) for i in range(101)],
            },
        ).status_code
        == 422
    )


def test_large_integer_offset_variance_and_csv_unit_errors_remain_exact():
    base = 10**50
    result = client.post(
        "/evaluate",
        json={
            "task": "regression",
            "actual": [base, base + 1],
            "predicted": [base + 1, base + 2],
        },
    ).json()["overall"]
    assert result["mae"] == 1 and result["rmse"] == 1
    assert result["r2"] == pytest.approx(-3)
    response = client.post(
        "/csv/evaluate",
        json={
            "task": "regression",
            "csv_text": f"actual,predicted\n{base},{base + 1}\n{base + 1},{base + 2}",
            "actual_column": "actual",
            "predicted_column": "predicted",
        },
    )
    assert response.status_code == 200
    overall = response.json()["overall"]
    assert overall["mae"] == 1 and overall["rmse"] == 1
    assert overall["r2"] == pytest.approx(-3)
