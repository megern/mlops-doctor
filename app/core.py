"""Manifest diagnostics; drift is an effect size, not a significance test."""

from typing import Any, Literal
from statistics import mean, pstdev
from math import isfinite
from pydantic import BaseModel, ConfigDict, Field


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train: list[dict[str, Any]] = Field(min_length=1, max_length=10000)
    validation: list[dict[str, Any]] = Field(min_length=1, max_length=10000)
    columns: dict[str, Literal["number", "string", "boolean"]] = Field(alias="schema")
    id_column: str
    target: str
    features: list[str]
    drift_threshold: float = Field(default=1.0, gt=0, allow_inf_nan=False)


def analyze(request: Request) -> dict:
    findings = []
    if (
        request.id_column not in request.columns
        or request.target not in request.columns
    ):
        raise ValueError("ID and target must be declared in schema")
    if any(f not in request.columns for f in request.features):
        raise ValueError("Every feature must be declared in schema")
    if request.target in request.features:
        findings.append(
            {
                "severity": "critical",
                "code": "TARGET_LEAKAGE",
                "detail": "Target is included in input features",
            }
        )

    def matches(value, kind):
        return (
            (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and abs(value) <= 1e100
                and isfinite(value)
            )
            if kind == "number"
            else isinstance(value, str)
            if kind == "string"
            else isinstance(value, bool)
        )

    for split, rows in [("train", request.train), ("validation", request.validation)]:
        for index, row in enumerate(rows):
            if set(row) != set(request.columns) or any(
                not matches(row.get(f), kind) for f, kind in request.columns.items()
            ):
                findings.append(
                    {
                        "severity": "error",
                        "code": "SCHEMA_MISMATCH",
                        "detail": f"{split} row {index} differs from schema",
                    }
                )

    def ids(rows):
        values = [r.get(request.id_column) for r in rows]
        if any(not isinstance(v, (str, int, float, bool)) for v in values):
            raise ValueError("IDs must be scalar, non-null values")
        return {(type(v).__name__, v) for v in values}

    train_ids, val_ids = ids(request.train), ids(request.validation)
    overlap = train_ids & val_ids
    if overlap:
        findings.append(
            {
                "severity": "critical",
                "code": "SPLIT_LEAKAGE",
                "detail": f"{len(overlap)} IDs occur in both splits",
            }
        )
    for split, rows, unique in [
        ("train", request.train, train_ids),
        ("validation", request.validation, val_ids),
    ]:
        if len(unique) != len(rows):
            findings.append(
                {
                    "severity": "warning",
                    "code": "DUPLICATE_ID",
                    "detail": f"{split} contains repeated IDs",
                }
            )
    drift = []
    for feature in request.features:
        if request.columns[feature] != "number":
            continue
        a, b = (
            [r.get(feature) for r in request.train],
            [r.get(feature) for r in request.validation],
        )
        if not all(matches(v, "number") for v in a + b):
            continue
        delta = abs(mean(a) - mean(b))
        if not isfinite(delta):
            raise ValueError("Numeric mean difference exceeds finite range")
        scale = pstdev(a)
        effect = delta / scale if scale else None
        if effect is not None and not isfinite(effect):
            raise ValueError(
                "Standardized mean shift exceeds finite range; rescale numeric inputs"
            )
        flagged = effect >= request.drift_threshold if effect is not None else delta > 0
        drift.append(
            {
                "feature": feature,
                "train_mean": mean(a),
                "validation_mean": mean(b),
                "train_std": scale,
                "standardized_mean_shift": effect,
                "constant_reference_changed": scale == 0 and delta > 0,
            }
        )
        if flagged:
            findings.append(
                {
                    "severity": "warning",
                    "code": "MEAN_DRIFT",
                    "detail": f"{feature} changed relative to training baseline",
                }
            )
    return {
        "healthy": not findings,
        "findings": findings,
        "numeric_drift": drift,
        "limitations": "ID overlap and target inclusion detect only explicit leakage. Mean drift does not detect all distribution shifts or establish statistical significance.",
    }
