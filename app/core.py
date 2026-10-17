"""Manifest diagnostics; drift is an effect size, not a significance test."""

from typing import Any, Literal
from statistics import mean, pstdev
from math import isfinite, fsum
from bisect import bisect_right
from collections import Counter
from pydantic import BaseModel, ConfigDict, Field


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train: list[dict[str, Any]] = Field(min_length=1, max_length=10000)
    validation: list[dict[str, Any]] = Field(min_length=1, max_length=10000)
    columns: dict[str, Literal["number", "string", "boolean"]] = Field(alias="schema")
    id_column: str
    target: str
    features: list[str] = Field(max_length=50)
    drift_threshold: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    distribution_checks: bool = False
    distribution_threshold: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)


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
    distributions = []
    if request.distribution_checks:
        for feature in dict.fromkeys(request.features):
            kind = request.columns[feature]
            a = [row.get(feature) for row in request.train]
            b = [row.get(feature) for row in request.validation]
            metric = "ks_cdf_distance" if kind == "number" else "total_variation"
            valid = all(matches(value, kind) for value in a + b)
            distance = (
                empirical_cdf_distance(a, b)
                if valid and kind == "number"
                else total_variation(a, b)
                if valid
                else None
            )
            flagged = distance is not None and distance > request.distribution_threshold
            distributions.append(
                {
                    "feature": feature,
                    "metric": metric,
                    "distance": distance,
                    "threshold": request.distribution_threshold,
                    "flagged": flagged,
                    "status": "computed" if valid else "skipped_invalid_values",
                    "train_count": len(a),
                    "validation_count": len(b),
                }
            )
            if flagged:
                findings.append(
                    {
                        "severity": "warning",
                        "code": "DISTRIBUTION_DRIFT",
                        "detail": f"{feature}: descriptive {metric} exceeds configured distance threshold",
                    }
                )
    return {
        "healthy": not findings,
        "findings": findings,
        "numeric_drift": drift,
        "distribution_checks_enabled": request.distribution_checks,
        "distribution_drift": distributions,
        "limitations": "ID overlap and target inclusion detect only explicit leakage. Mean drift does not detect all distribution shifts. Optional CDF/total-variation distances are descriptive, have no p-values or significance claims, and do not establish model degradation.",
    }


def empirical_cdf_distance(a, b):
    """Two-sample empirical CDF sup distance with exact tie handling; no p-value."""
    left, right = sorted(a), sorted(b)
    return max(
        abs(
            bisect_right(left, value) / len(left)
            - bisect_right(right, value) / len(right)
        )
        for value in set(left + right)
    )


def total_variation(a, b):
    left, right = Counter(a), Counter(b)
    return min(
        1.0,
        max(
            0.0,
            fsum(
                abs(left[value] / len(a) - right[value] / len(b))
                for value in left.keys() | right.keys()
            )
            / 2,
        ),
    )
