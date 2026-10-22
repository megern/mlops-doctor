# MLOps Doctor · 0.4.0

**Paired-split diagnostic workbench.** A practical validation tool for tabular training and validation manifests. It checks exact schema conformance, duplicate/overlapping IDs, accidental target inclusion and standardized numeric mean shifts.

The interface is a task-specific workbench, with labeled forms or CSV imports, previews, understandable diagnostics, charts/tables, actionable errors and reusable exports. It is a self-hosted engineering tool, not a claim of certified enterprise production readiness. Methods are deterministic and transparent; no paid service or GPU is required.

## Quickstart

```sh
docker compose up --build
```

Open http://127.0.0.1:8000. Click **Load synthetic example**, then **Run analysis**. Bundled data are explicitly synthetic. The full user guide is linked in the application at `/guide` and saved in [docs/USER_GUIDE.md](docs/USER_GUIDE.md). API schemas and examples are at `/docs`.

Run without Docker (Python 3.12):

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install --require-hashes -r requirements-dev.lock
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

For several apps, change the compose host port; each container listens on 8000. Stateless imports are processed in memory. Reports/configurations can contain your data and should be handled accordingly.

## Real functionality

Exact schema checks flag missing/extra fields and wrong types. Blank CSV cells become null. Numeric conversion is explicit; invalid text is retained and diagnosed as a schema mismatch. Booleans accept true/false; booleans are not numbers. IDs preserve scalar types in the API and use the selected CSV schema in imports.

SPLIT_LEAKAGE checks ID overlap; DUPLICATE_ID checks repeated IDs within each split; TARGET_LEAKAGE flags a target included in the feature list. These cannot detect near duplicates, label-derived feature leakage or temporal leakage automatically.

Numeric mean drift is absolute validation/train mean difference divided by the training population standard deviation. A changed constant baseline is flagged with a null standardized shift. Unusable numeric features are skipped, not imputed. Mean shift is an effect-size diagnostic, not a significance test or a model-performance estimate. Warnings also make healthy false.

## Workflow and examples

1. Upload or paste both training and validation CSV files. Preview them together.
2. Review each expected column type inferred from the training preview. ID-like names default to string to preserve identifiers such as 001.
3. Choose a record ID and target column. Select the actual input features. The target checkbox can deliberately reproduce accidental target leakage for diagnosis.
4. Choose the mean-drift threshold in training population standard deviations; the default is 1.
5. Run analysis. Read critical leakage findings before investigating warnings. Review typed schema errors before interpreting skipped numeric drift calculations.
6. Export the paired-data configuration and dated diagnostic report, or print the rendered results. Configurations contain both CSV texts; handle them as data exports.

The typed JSON API remains supported:

```sh
curl --fail-with-body http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' --data-binary @examples/synthetic.json
```

CSV-based applications also expose `/csv/preview` and `/csv/analyze`, documented in `/docs`. `examples/synthetic-builder.json` is a complete CSV endpoint request; use it with `/csv/analyze`. Importing CSV never executes spreadsheet formulas. Exported CSV from FlowPilot neutralizes formula-like text. Report/configuration export is JSON; browser print can save the result to PDF. No report is represented as an independent audit certification.

## Validation and evidence

```sh
python -m pytest -q
node --test tests/ui.test.mjs
python -m bandit -r app -ll
python -m pip_audit --disable-pip --no-deps --require-hashes -r requirements.lock
```

Historical 0.2 validation is retained under `evidence/rebuild/`; current 0.3 checks are recorded under `evidence/improvements-0.3/`. Older evidence documents the earlier implementation and must not be treated as proof of this interface. Meaningful existing arithmetic/security tests are retained. CI checks Python and frontend tests, dependency vulnerabilities, secret patterns, container vulnerabilities and real HTTP smoke operation. Require the `quality` check in repository branch protection to prevent merging failing revisions; a workflow alone cannot enforce that policy.

## Limits and troubleshooting

The synthetic example has one overlapping ID and a large mean shift; it should raise SPLIT_LEAKAGE and MEAN_DRIFT. A quiet report does not prove an ML pipeline is healthy.

For schema errors, compare column names, missing cells and the chosen types. CSV order does not affect matching, but extra/missing names do. Missing or nonscalar IDs prevent meaningful leakage detection and produce a validation error. Very small training variability may overflow the standardized ratio; rescale the inputs if instructed. Optional numeric and categorical distribution checks are available when enabled; prediction metrics are available in the separate evaluation panel. Fairness certification and trained-model inspection are not implemented.

HTTP JSON requests are bounded at 2 MiB; CSV parsing is bounded at 50 columns/10000 rows. Requests are local CPU work, no outside actions. Authentication and HTTPS are required when deploying beyond a trusted localhost environment. See the user guide for detailed method/resource limitations. This release provides optional shared-key API access and per-process compute admission. It does not provide durable history or multi-user access management; see docs/OPERATIONS.md.

## License

MIT. See LICENSE. Review transitive dependency licenses for your distribution. Contributors and review evidence are automated engineering work; no human reviewer is invented.

## Optional distribution checks in 0.3

Enable the explicit distribution-check checkbox to compare every selected feature's empirical distributions. It is off by default so existing API callers retain prior diagnostic behavior. Set a descriptive distance threshold from 0 to 1; the default is 0.25. A warning is raised only when distance is strictly greater than the threshold. Thus identical distributions are not flagged at zero, and a threshold of one suppresses distance flags.

Numeric features use the maximum difference between empirical cumulative distribution functions (a two-sample KS distance), with ties counted together. For train [-1, 1] and validation [-2, 2], both means are zero but the distance is 0.5, revealing a shape change that a mean-only check misses. There are no p-values, significance levels, hypothesis-test conclusions or sample-size calibration.

String and boolean features use total variation: half the sum of absolute differences in category proportions. Train [A,A,B,B] and validation [A,A,A,B] have distance 0.25. Categories use exact values and no smoothing. These descriptive metrics depend on sample representativeness and cannot prove model degradation, fairness issues, causal changes or distribution equivalence.

A feature with missing or invalid cells is skipped for its distribution calculation and explicitly labeled “skipped_invalid_values”; existing schema diagnostics still identify errors. No imputation or silent record removal is performed. API fields are distribution_checks (default false) and distribution_threshold (0–1), supported by both /analyze and /csv/analyze. Result distribution_drift rows report metric, distance, threshold, status, sample counts and flag. Configuration/report exports retain these settings and results. The API allows at most 50 selected features; CSV import remains bounded at 50 columns/10000 records.

Fresh 0.3 validation is saved in `evidence/improvements-0.3/`; older reports describe earlier versions and are not proof of these additions. This is automated developer work with separate release-manager and independent QA gates; no human review or enterprise certification is claimed.

The new synthetic request `examples/synthetic-improvements-03.json` demonstrates this version. POST it to `/csv/analyze`; all data are synthetic.

## Prediction evaluation and slices in 0.4

The Prediction evaluation panel is separate from training/validation pipeline diagnostics. Upload or paste a held-out predictions CSV, preview it, select classification or regression, and map distinct actual, predicted and optional group columns. Set minimum rows per group (1–10,000, default 5). Load synthetic prediction evaluation supplies six fictional classification rows; expected accuracy is 4/6, with two groups of three. It never represents real people or a fairness study.

Classification shows accuracy, a confusion matrix (rows actual, columns predicted), class support, precision, recall and F1, plus macro and support-weighted averages. The label universe is the union of observed actual and predicted labels across the full input, reused for every slice. Thus a globally observed class absent in a slice still contributes zero to that slice's macro average. Precision = TP / predicted class count; recall = TP / actual class count; F1 is their harmonic mean. A zero denominator yields 0. Weighted averages use actual support; macro gives every label equal weight. At most 50 distinct labels are permitted. CSV classification labels are exact case-sensitive text after trimming; direct JSON supports distinct boolean, numeric and text labels, with numeric 1 and 1.0 equivalent.

Regression shows MAE = mean absolute error, RMSE = square root of mean squared error and R² = 1−SSE/SST about the actual mean. R² may be negative; it is null with a reason for constant actual values or a ratio outside finite floating-point range. Scaled square calculations protect very small error magnitudes from avoidable underflow. Regression requires finite numbers with magnitude at most 1e100; booleans, missing values and numeric-looking text in direct JSON are rejected. CSV mappings explicitly convert numeric strings for regression.

The overall result uses every row. Group names must be nonblank text of at most 200 characters, with at most 100 distinct groups. Below-minimum-support groups report their support and status but no metrics. Supported slices report the same metric definitions as overall. There is no confidence interval, p-value, significance test, fairness certification or causal attribution. Labels, slice definitions, missing groups, selection bias and representativeness require domain review. No rows are silently dropped or imputed.

POST /evaluate accepts task, actual, predicted, optional groups and minimum_support. POST /csv/evaluate accepts csv_text, task, actual_column, predicted_column, optional group_column and minimum_support. Arrays and CSV are capped at 10,000 rows and the shared 2 MiB request bound applies. Prediction report and configuration exports include these inputs and outputs; keep them private if labels/group names are sensitive. Existing /analyze and /csv/analyze pipeline behavior and distribution-check defaults remain unchanged. See examples/synthetic-predictions.json and examples/synthetic-predictions.csv. If imports fail, map distinct existing columns and inspect blank cells; classification requires nonempty labels, regression requires numbers.

Developer validation for 0.4 is saved in `evidence/improvements-0.4/`. Prior evidence describes prior versions. Local developer tests do not establish independent acceptance, hosted CI success or deployment readiness.

## Server access and operating limits

See [Operations](docs/OPERATIONS.md) for optional shared-key API access, per-process request admission, platform notes and the additional work needed for public multi-user hosting.
