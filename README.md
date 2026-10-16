# MLOps Doctor · 0.2.0

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

New 0.2 validation is recorded under `evidence/rebuild/`. Older evidence documents the earlier implementation and must not be treated as proof of this interface. Meaningful existing arithmetic/security tests are retained. CI checks Python and frontend tests, dependency vulnerabilities, secret patterns, container vulnerabilities and real HTTP smoke operation. Require the `quality` check in repository branch protection to prevent merging failing revisions; a workflow alone cannot enforce that policy.

## Limits and troubleshooting

The synthetic example has one overlapping ID and a large mean shift; it should raise SPLIT_LEAKAGE and MEAN_DRIFT. A quiet report does not prove an ML pipeline is healthy.

For schema errors, compare column names, missing cells and the chosen types. CSV order does not affect matching, but extra/missing names do. Missing or nonscalar IDs prevent meaningful leakage detection and produce a validation error. Very small training variability may overflow the standardized ratio; rescale the inputs if instructed. Categorical distribution drift, fairness metrics, model metrics and trained-model inspection are not implemented.

HTTP JSON requests are bounded at 2 MiB; CSV parsing is bounded at 50 columns/10000 rows. Requests are local CPU work, no outside actions. Authentication and HTTPS are required when deploying beyond a trusted localhost environment. See the user guide for detailed method/resource limitations. This release provides no authentication, durable history or multi-user access management.

## License

MIT. See LICENSE. Review transitive dependency licenses for your distribution. Contributors and review evidence are automated engineering work; no human reviewer is invented.
