"""Descriptive prediction metrics with bounded, explicitly supported slices."""

from math import sqrt, isfinite, fsum
from decimal import Decimal, InvalidOperation
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from .csv_import import parse_csv


class Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["classification", "regression"]
    actual: list[str | int | float | bool] = Field(min_length=1, max_length=10000)
    predicted: list[str | int | float | bool] = Field(min_length=1, max_length=10000)
    groups: list[str] | None = Field(default=None, max_length=10000)
    minimum_support: int = Field(default=5, ge=1, le=10000)


class CSVEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    csv_text: str = Field(min_length=1, max_length=2097152)
    task: Literal["classification", "regression"]
    actual_column: str
    predicted_column: str
    group_column: str | None = None
    minimum_support: int = Field(default=5, ge=1, le=10000)


def numeric_difference(left, right):
    # Integer subtraction retains unit differences beyond binary64 precision.
    if isinstance(left, int) and isinstance(right, int):
        return left - right
    if isinstance(left, int) or isinstance(right, int):
        return float(Decimal(left) - Decimal(right))
    return left - right


def token(value):
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, float)):
        if not isfinite(value) or abs(value) > 1e100:
            raise ValueError("Labels must be finite and bounded")
        return ("number", value)
    if not value.strip() or len(value) > 200:
        raise ValueError("Labels must be nonblank and at most 200 characters")
    return ("text", value)


def evaluate(request: Evaluation):
    if (
        len(request.actual) != len(request.predicted)
        or request.groups is not None
        and len(request.groups) != len(request.actual)
    ):
        raise ValueError("Actual, predicted and groups must have equal lengths")
    if request.task == "classification":
        keys = sorted(
            {token(v) for v in request.actual + request.predicted},
            key=lambda t: (t[0], str(t[1])),
        )
        if len(keys) > 50:
            raise ValueError("Classification is limited to 50 distinct labels")
        lookup = {k: i for i, k in enumerate(keys)}

        def metric(indices):
            matrix = [[0] * len(keys) for _ in keys]
            for i in indices:
                matrix[lookup[token(request.actual[i])]][
                    lookup[token(request.predicted[i])]
                ] += 1
            support = [sum(row) for row in matrix]
            n = len(indices)
            per = []
            for j, k in enumerate(keys):
                tp = matrix[j][j]
                pred = sum(row[j] for row in matrix)
                precision = tp / pred if pred else 0
                recall = tp / support[j] if support[j] else 0
                per.append(
                    {
                        "label": k[1],
                        "label_type": k[0],
                        "support": support[j],
                        "precision": precision,
                        "recall": recall,
                        "f1": 2 * precision * recall / (precision + recall)
                        if precision + recall
                        else 0,
                    }
                )
            return {
                "support": n,
                "accuracy": sum(matrix[i][i] for i in range(len(keys))) / n,
                "labels": [{"type": k[0], "value": k[1]} for k in keys],
                "confusion_matrix": matrix,
                "per_class": per,
                "macro": {
                    m: fsum(row[m] for row in per) / len(keys)
                    for m in ["precision", "recall", "f1"]
                },
                "weighted": {
                    m: fsum(row[m] * row["support"] for row in per) / n
                    for m in ["precision", "recall", "f1"]
                },
            }
    else:
        for value in request.actual + request.predicted:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or abs(value) > 1e100
            ):
                raise ValueError(
                    "Regression needs finite numeric values of magnitude at most 1e100"
                )

        def metric(indices):
            actual = [request.actual[i] for i in indices]
            errors = [
                numeric_difference(request.actual[i], request.predicted[i])
                for i in indices
            ]
            n = len(indices)
            offsets = [numeric_difference(v, actual[0]) for v in actual]
            offset_scale = max(abs(v) for v in offsets)
            normalized = (
                [v / offset_scale for v in offsets] if offset_scale else [0.0] * n
            )
            normalized_mean = fsum(normalized) / n
            error_scale = max(abs(e) for e in errors)
            deviation_scale = offset_scale
            scaled_errors = (
                fsum((e / error_scale) ** 2 for e in errors) if error_scale else 0
            )
            rmse = sqrt(scaled_errors / n) * error_scale
            reason = None
            r2 = None
            if deviation_scale == 0:
                reason = "constant actual values"
            else:
                scaled_deviations = fsum((v - normalized_mean) ** 2 for v in normalized)
                try:
                    ratio = (
                        (error_scale / deviation_scale) ** 2
                        * scaled_errors
                        / scaled_deviations
                    )
                    if isfinite(ratio):
                        r2 = 1 - ratio
                    else:
                        reason = "R2 exceeds finite numeric range"
                except OverflowError:
                    reason = "R2 exceeds finite numeric range"
            return {
                "support": n,
                "mae": fsum(abs(e) for e in errors) / n,
                "rmse": rmse,
                "r2": r2,
                "r2_undefined_reason": reason,
            }

    overall = metric(list(range(len(request.actual))))
    groups = {}
    if request.groups is not None:
        for i, value in enumerate(request.groups):
            if not value.strip() or len(value) > 200:
                raise ValueError(
                    "Group names must be nonblank and at most 200 characters"
                )
            groups.setdefault(value, []).append(i)
            if len(groups) > 100:
                raise ValueError("At most 100 distinct groups")
    slices = []
    for group, indices in sorted(groups.items()):
        slices.append(
            {
                "group": group,
                "support": len(indices),
                "status": "evaluated"
                if len(indices) >= request.minimum_support
                else "below_minimum_support",
                "metrics": metric(indices)
                if len(indices) >= request.minimum_support
                else None,
            }
        )
    return {
        "task": request.task,
        "overall": overall,
        "slices": slices,
        "minimum_support": request.minimum_support,
        "method": "Descriptive predictions only. Confusion matrix rows=actual, columns=predicted. Global observed/predicted label universe is used in every slice. Undefined precision/recall/F1 denominators yield 0; macro is unweighted over that universe; weighted uses actual support. Regression R2=1-SSE/SST and is null for constant actual values or unrepresentable numeric ratios (reason supplied). Slice comparisons do not establish fairness, causal effects or statistical significance.",
    }


def evaluate_csv(request: CSVEvaluation):
    headers, rows = parse_csv(request.csv_text)
    mapping = [request.actual_column, request.predicted_column] + (
        [request.group_column] if request.group_column else []
    )
    if len(set(mapping)) != len(mapping) or any(c not in headers for c in mapping):
        raise ValueError(
            "Map distinct existing columns for actual, predicted and optional group"
        )

    def values(column):
        raw = [row[column].strip() for row in rows]
        if request.task == "classification":
            return raw
        converted = []
        for cell in raw:
            try:
                value = Decimal(cell)
                if not value.is_finite() or abs(value) > Decimal("1e100"):
                    raise ValueError(
                        "Regression values must be finite with magnitude at most 1e100"
                    )
                converted.append(
                    int(value) if value == value.to_integral_value() else float(value)
                )
                if value and not converted[-1]:
                    raise ValueError(
                        "Regression value is below finite floating-point precision; rescale input"
                    )
            except (InvalidOperation, ValueError) as exc:
                raise ValueError(
                    "Regression columns need bounded numeric cells with no empty values: "
                    + str(exc)
                ) from exc
        return converted

    return evaluate(
        Evaluation(
            task=request.task,
            actual=values(request.actual_column),
            predicted=values(request.predicted_column),
            groups=[row[request.group_column].strip() for row in rows]
            if request.group_column
            else None,
            minimum_support=request.minimum_support,
        )
    )
