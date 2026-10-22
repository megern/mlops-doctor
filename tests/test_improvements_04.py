import math
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.evaluation import Evaluation, evaluate

client = TestClient(app)


def test_classification_confusion_macro_weighted_and_minimum_support():
    r = evaluate(
        Evaluation(
            task="classification",
            actual=["a", "a", "b", "b", "b"],
            predicted=["a", "b", "b", "b", "a"],
            groups=["one", "one", "two", "two", "two"],
            minimum_support=3,
        )
    )
    assert r["overall"]["confusion_matrix"] == [[1, 1], [1, 2]]
    assert r["overall"]["accuracy"] == 0.6
    assert r["overall"]["macro"]["f1"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert r["overall"]["weighted"]["f1"] == pytest.approx((0.5 * 2 + 2 / 3 * 3) / 5)
    assert (
        r["slices"][0]["status"] == "below_minimum_support"
        and r["slices"][0]["metrics"] is None
    )
    assert r["slices"][1]["metrics"]["accuracy"] == pytest.approx(2 / 3)


def test_regression_known_numbers_and_constant_target():
    r = evaluate(
        Evaluation(
            task="regression",
            actual=[1, 2, 3],
            predicted=[1, 3, 2],
            groups=["a", "a", "b"],
            minimum_support=1,
        )
    )
    assert r["overall"]["mae"] == pytest.approx(2 / 3)
    assert r["overall"]["rmse"] == pytest.approx(math.sqrt(2 / 3))
    assert r["overall"]["r2"] == 0
    assert r["slices"][1]["metrics"]["r2"] is None


def test_typed_classification_labels_stay_distinct_and_zero_denominators():
    r = evaluate(
        Evaluation(
            task="classification", actual=[True, 1, "1"], predicted=["1", "1", "1"]
        )
    )
    assert len(r["overall"]["labels"]) == 3
    assert r["overall"]["accuracy"] == pytest.approx(1 / 3)
    assert r["overall"]["macro"]["recall"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "data",
    [
        {"actual": [1, 2], "predicted": [1]},
        {"actual": [1], "predicted": [1], "groups": ["a", "b"]},
        {"actual": [True], "predicted": [1], "task": "regression"},
        {"actual": [""], "predicted": ["a"]},
        {"actual": list(range(51)), "predicted": list(range(51))},
        {
            "actual": [1] * 101,
            "predicted": [1] * 101,
            "groups": [str(i) for i in range(101)],
        },
        {"actual": [None], "predicted": [1]},
        {"actual": [1], "predicted": [1], "minimum_support": 0},
    ],
)
def test_evaluation_input_errors_are_422(data):
    assert (
        client.post(
            "/evaluate",
            json={"task": "classification", "actual": [1], "predicted": [1], **data},
        ).status_code
        == 422
    )


def test_csv_classification_mapper_and_empty_record_validation():
    data = {
        "csv_text": "true,predicted,group\nyes,yes,A\nno,yes,B",
        "task": "classification",
        "actual_column": "true",
        "predicted_column": "predicted",
        "group_column": "group",
        "minimum_support": 1,
    }
    r = client.post("/csv/evaluate", json=data)
    assert r.status_code == 200 and r.json()["overall"]["accuracy"] == 0.5
    data["csv_text"] = "true,predicted,group\n,,"
    assert client.post("/csv/evaluate", json=data).status_code == 422
    data["csv_text"] = "true,predicted,group\n1,2,A"
    data["task"] = "regression"
    assert client.post("/csv/evaluate", json=data).json()["overall"]["mae"] == 1


def test_regression_extreme_ratios_never_serialize_infinity_and_small_rmse_survives():
    response = client.post(
        "/evaluate",
        json={"task": "regression", "actual": [0, 1e-200], "predicted": [1e100, 1e100]},
    )
    assert response.status_code == 200
    assert response.json()["overall"]["r2"] is None
    assert (
        response.json()["overall"]["r2_undefined_reason"]
        == "R2 exceeds finite numeric range"
    )
    r = evaluate(
        Evaluation(task="regression", actual=[0, 1e-200], predicted=[1e-200, 0])
    )
    assert (
        r["overall"]["rmse"] == pytest.approx(1e-200, abs=0)
        and r["overall"]["r2"] == -3
    )


def test_large_integer_evaluation_preserves_labels_differences_and_centered_variance():
    a = 9007199254740992
    b = a + 1
    assert (
        client.post(
            "/evaluate",
            json={"task": "classification", "actual": [a], "predicted": [b]},
        ).json()["overall"]["accuracy"]
        == 0
    )
    r = client.post(
        "/evaluate", json={"task": "regression", "actual": [a, b], "predicted": [b, a]}
    ).json()["overall"]
    assert r["mae"] == r["rmse"] == 1 and r["r2"] == -3
    r = client.post(
        "/evaluate", json={"task": "regression", "actual": [b], "predicted": [float(a)]}
    ).json()["overall"]
    assert r["mae"] == 1
    r = client.post(
        "/csv/evaluate",
        json={
            "task": "regression",
            "csv_text": f"a,p\n{a},{b}",
            "actual_column": "a",
            "predicted_column": "p",
        },
    ).json()["overall"]
    assert r["mae"] == 1


def test_subnormal_actual_variance_and_rmse_are_scaled_before_centering():
    tiny = 5e-324
    r = client.post(
        "/evaluate",
        json={"task": "regression", "actual": [0, tiny], "predicted": [tiny, 0]},
    ).json()["overall"]
    assert r["mae"] == tiny and r["rmse"] == tiny and r["r2"] == -3
