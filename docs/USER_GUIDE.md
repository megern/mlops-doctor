# MLOps Doctor — User guide

## Intended use

A practical validation tool for tabular training and validation manifests. It checks exact schema conformance, duplicate/overlapping IDs, accidental target inclusion and standardized numeric mean shifts.

## Walkthrough

1. Upload or paste both training and validation CSV files. Preview them together.
2. Review each expected column type inferred from the training preview. ID-like names default to string to preserve identifiers such as 001.
3. Choose a record ID and target column. Select the actual input features. The target checkbox can deliberately reproduce accidental target leakage for diagnosis.
4. Choose the mean-drift threshold in training population standard deviations; the default is 1.
5. Run analysis. Read critical leakage findings before investigating warnings. Review typed schema errors before interpreting skipped numeric drift calculations.
6. Export the paired-data configuration and dated diagnostic report, or print the rendered results. Configurations contain both CSV texts; handle them as data exports.

## Method and interpretation

Exact schema checks flag missing/extra fields and wrong types. Blank CSV cells become null. Numeric conversion is explicit; invalid text is retained and diagnosed as a schema mismatch. Booleans accept true/false; booleans are not numbers. IDs preserve scalar types in the API and use the selected CSV schema in imports.

SPLIT_LEAKAGE checks ID overlap; DUPLICATE_ID checks repeated IDs within each split; TARGET_LEAKAGE flags a target included in the feature list. These cannot detect near duplicates, label-derived feature leakage or temporal leakage automatically.

Numeric mean drift is absolute validation/train mean difference divided by the training population standard deviation. A changed constant baseline is flagged with a null standardized shift. Unusable numeric features are skipped, not imputed. Mean shift is an effect-size diagnostic, not a significance test or a model-performance estimate. Warnings also make healthy false.

## Examples and troubleshooting

The synthetic example has one overlapping ID and a large mean shift; it should raise SPLIT_LEAKAGE and MEAN_DRIFT. A quiet report does not prove an ML pipeline is healthy.

For schema errors, compare column names, missing cells and the chosen types. CSV order does not affect matching, but extra/missing names do. Missing or nonscalar IDs prevent meaningful leakage detection and produce a validation error. Very small training variability may overflow the standardized ratio; rescale the inputs if instructed. Optional numeric and categorical distribution checks are available when enabled; fairness metrics, model metrics and trained-model inspection are not implemented.

## Resource and privacy limits

The HTTP guard bounds state-changing JSON bodies at 2 MiB and rejects excessive nesting, nonfinite numbers and numeric magnitudes beyond 1e100. CSV imports allow up to 50 unique columns and 10000 data records, with 10000-character cells. FlowPilot additionally caps each node at 10000 rows and conservatively bounds retained workflow outputs at 4 MiB. Browser previews and tables are bounded. Splitting files can change the analytical meaning: do not split event cases or evaluation splits arbitrarily.

The application is stateless and does not persist uploads. Your browser may retain downloads and form content; printed reports and configurations can contain the input data. No external resources or paid APIs are contacted. Requests go to the server you opened: self-host on a trusted machine. Authentication, multi-user access controls, secure ingress, rate limiting and operational backups are deployment responsibilities, not claims of this release.

## API and reuse

Open /docs for the complete schema. POST /analyze remains available for typed JSON inputs. CSV-based tools additionally provide POST /csv/preview and POST /csv/analyze. Configuration exports map directly to the relevant endpoint; readiness uses /analyze, the three CSV tools use /csv/analyze. Report exports contain application/version/date, synthetic provenance, original input and analytical output. They are data files, not executable programs.

All bundled examples are synthetic and are labeled. Editing or importing your own input resets that badge and clears stale results. Controls are locked during analysis; a generation check also prevents a late response replacing changed input results. Error messages identify the failed field or operation; correct it and rerun. Failed analyses never certify completion or preserve an old result as current.

## Local operation

Run docker compose up --build and open http://127.0.0.1:8000. Alternatively install requirements-dev.lock with hashes in a Python 3.12 environment and run uvicorn app.main:app --host 127.0.0.1 --port 8000. Change the host port to run multiple tools. No GPU is needed.

## Optional distribution checks in 0.3

Enable the explicit distribution-check checkbox to compare every selected feature's empirical distributions. It is off by default so existing API callers retain prior diagnostic behavior. Set a descriptive distance threshold from 0 to 1; the default is 0.25. A warning is raised only when distance is strictly greater than the threshold. Thus identical distributions are not flagged at zero, and a threshold of one suppresses distance flags.

Numeric features use the maximum difference between empirical cumulative distribution functions (a two-sample KS distance), with ties counted together. For train [-1, 1] and validation [-2, 2], both means are zero but the distance is 0.5, revealing a shape change that a mean-only check misses. There are no p-values, significance levels, hypothesis-test conclusions or sample-size calibration.

String and boolean features use total variation: half the sum of absolute differences in category proportions. Train [A,A,B,B] and validation [A,A,A,B] have distance 0.25. Categories use exact values and no smoothing. These descriptive metrics depend on sample representativeness and cannot prove model degradation, fairness issues, causal changes or distribution equivalence.

A feature with missing or invalid cells is skipped for its distribution calculation and explicitly labeled “skipped_invalid_values”; existing schema diagnostics still identify errors. No imputation or silent record removal is performed. API fields are distribution_checks (default false) and distribution_threshold (0–1), supported by both /analyze and /csv/analyze. Result distribution_drift rows report metric, distance, threshold, status, sample counts and flag. Configuration/report exports retain these settings and results. The API allows at most 50 selected features; CSV import remains bounded at 50 columns/10000 records.
