import {
  getRevision,
  isStale,
  assertCurrent,
  $,
  el,
  replace,
  table,
  previewRows,
  bars,
  metrics,
  api,
  action,
  status,
  newReport,
  setupExports,
  invalidate,
  bindFile,
  readFile,
  schemaEditor,
  setOptions,
  selectField,
  download,
} from "./common.js";
let columns = [],
  getSchema = () => ({}),
  synthetic = false,
  report = null;
function dirty() {
  synthetic = false;
  report = null;
  $("data-origin").textContent = "Your input · changed";
  $("technical").textContent = "Inputs changed. Run again.";
  invalidate();
}
for (const id of ["train", "validation", "threshold", "distribution-threshold"])
  $(id).oninput = dirty;
$("schema").onchange = dirty;
$("distribution-checks").onchange = dirty;
function features(selected) {
  replace($("features"));
  for (const column of columns) {
    const label = el("label"),
      checkbox = el("input");
    checkbox.type = "checkbox";
    checkbox.value = column;
    checkbox.checked = selected
      ? selected.includes(column)
      : column !== $("id-column").value && column !== $("target").value;
    checkbox.onchange = dirty;
    label.append(
      checkbox,
      el(
        "span",
        column +
          (column === $("target").value
            ? " · target (including it will flag leakage)"
            : ""),
      ),
    );
    $("features").append(label);
  }
}
for (const id of ["id-column", "target"])
  $(id).onchange = () => {
    features();
    dirty();
  };
async function preview() {
  report = null;
  invalidate();
  const generation = getRevision();
  const [train, validation] = await Promise.all([
    api("/csv/preview", { csv_text: $("train").value }),
    api("/csv/preview", { csv_text: $("validation").value }),
  ]);
  assertCurrent(generation);
  columns = train.columns;
  replace(
    $("preview-data"),
    el("h3", "Training · " + train.row_count + " rows"),
    previewRows(train.columns, train.preview),
    el("h3", "Validation · " + validation.row_count + " rows"),
    previewRows(validation.columns, validation.preview),
  );
  getSchema = schemaEditor($("schema"), columns, train.preview);
  setOptions(
    $("id-column"),
    columns,
    columns.find((c) => /^(id|record_id)$/i.test(c)) || columns[0],
  );
  setOptions(
    $("target"),
    columns,
    columns.find((c) => /^(y|target|label)$/i.test(c)) || columns.at(-1),
  );
  features();
  if (train.columns.join("|") !== validation.columns.join("|"))
    status(
      "Column sets or order differ. Extra/missing fields are diagnosed against the chosen training schema; CSV column order itself is not an error.",
    );
  return { train, validation, generation };
}
$("preview").onclick = async () => {
  try {
    dirty();
    const prepared = await preview();
    assertCurrent(prepared.generation);
    status(
      "Files parsed. Review types, ID, target and selected features before diagnosis.",
    );
  } catch (e) {
    if (isStale(e)) return;
    status(e.message, true);
  }
};
for (const [field, file] of [
  ["train", "train-file"],
  ["validation", "validation-file"],
])
  bindFile(
    file,
    field,
    async () => {
      dirty();
      status("CSV loaded. Import the other split, then preview both files.");
    },
    dirty,
  );
function config() {
  return {
    train_csv: $("train").value,
    validation_csv: $("validation").value,
    schema: getSchema(),
    id_column: $("id-column").value,
    target: $("target").value,
    features: [...$("features").querySelectorAll("input:checked")].map(
      (n) => n.value,
    ),
    drift_threshold: Number($("threshold").value),
    distribution_checks: $("distribution-checks").checked,
    distribution_threshold: Number($("distribution-threshold").value),
  };
}
$("example").onclick = async () => {
  try {
    dirty();
    $("train").value = "id,x,y\na,1,0\nb,3,1";
    $("validation").value = "id,x,y\na,8,1\nc,10,0";
    $("threshold").value = 1;
    $("distribution-checks").checked = false;
    $("distribution-threshold").value = 0.25;
    const prepared = await preview();
    assertCurrent(prepared.generation);
    synthetic = true;
    $("data-origin").textContent =
      "Synthetic example · no real training records";
    status(
      "Synthetic example includes one overlapping ID and a large numeric mean shift.",
    );
  } catch (e) {
    if (isStale(e)) return;
    status(e.message, true);
  }
};
$("config-file").onchange = async () => {
  try {
    dirty();
    const generation = getRevision();
    const text = await readFile($("config-file"));
    assertCurrent(generation);
    if (text === null) return;
    const data = JSON.parse(text);
    if (
      typeof data.train_csv !== "string" ||
      typeof data.validation_csv !== "string" ||
      !data.schema ||
      !Array.isArray(data.features)
    )
      throw new Error("Use a diagnosis configuration exported by this tool.");
    $("train").value = data.train_csv;
    $("validation").value = data.validation_csv;
    dirty();
    const prepared = await preview();
    assertCurrent(prepared.generation);
    for (const [c, t] of Object.entries(data.schema)) {
      const s = document.getElementById("type-" + c);
      if (s && ["number", "boolean", "string"].includes(t)) s.value = t;
    }
    setOptions($("id-column"), columns, data.id_column);
    setOptions($("target"), columns, data.target);
    features(data.features);
    $("threshold").value = data.drift_threshold ?? 1;
    $("distribution-checks").checked = data.distribution_checks === true;
    $("distribution-threshold").value = data.distribution_threshold ?? 0.25;
    status(
      "Diagnosis configuration imported. Review the schema before running.",
    );
  } catch (e) {
    if (isStale(e)) return;
    dirty();
    status("Cannot import configuration: " + e.message, true);
  }
};
const actions = {
  DISTRIBUTION_DRIFT:
    "Inspect the feature distribution and collection process. Distances can flag equal-mean shape changes; they provide no p-value or proof of performance degradation.",
  TARGET_LEAKAGE:
    "Remove the target from input features; review upstream feature construction for derived-label leakage.",
  SPLIT_LEAKAGE:
    "Identify overlapping record IDs and rebuild disjoint splits according to your evaluation strategy.",
  DUPLICATE_ID:
    "Investigate repeated records and whether the identifier uniquely represents a sample.",
  SCHEMA_MISMATCH:
    "Correct missing/extra columns or invalid cell types. This tool does not impute missing values.",
  MEAN_DRIFT:
    "Investigate data collection changes and plot feature distributions. A mean shift alone does not prove performance degradation.",
};
$("run").onclick = () =>
  action(async () => {
    report = null;
    const input = config();
    if (!columns.length)
      throw new Error(
        "Preview both CSV files and define the expected schema first.",
      );
    const generation = getRevision(),
      result = await api("/csv/analyze", input);
    assertCurrent(generation);
    report = newReport("MLOps Doctor", input, result, synthetic);
    const counts = result.findings.reduce(
      (a, f) => ((a[f.severity] = (a[f.severity] || 0) + 1), a),
      {},
    );
    replace(
      $("results"),
      metrics([
        ["Findings", result.findings.length],
        ["Critical", counts.critical || 0],
        ["Schema errors", counts.error || 0],
      ]),
      el(
        "p",
        result.healthy
          ? "No configured diagnostic flags. This does not prove absence of leakage or drift."
          : "Review the flagged checks before trusting this evaluation.",
        "notice",
      ),
    );
    if (result.findings.length) {
      $("results").append(
        table(
          ["Severity", "Check", "Detail"],
          result.findings.map((f) => [f.severity, f.code, f.detail]),
        ),
      );
      for (const code of new Set(result.findings.map((f) => f.code))) {
        const card = el("div", "", "step");
        card.append(
          el("strong", code),
          el(
            "p",
            actions[code] ||
              "Review the diagnostic and verify the source data.",
          ),
        );
        $("results").append(card);
      }
    }
    if (result.numeric_drift.length) {
      $("results").append(el("h3", "Standardized numeric mean shifts"));
      const finite = result.numeric_drift.filter(
        (d) => d.standardized_mean_shift !== null,
      );
      $("results").append(
        bars(
          finite.map((d) => [d.feature, d.standardized_mean_shift]),
          Math.max(
            input.drift_threshold,
            ...finite.map((d) => d.standardized_mean_shift),
          ),
        ),
        table(
          ["Feature", "Train mean", "Validation mean", "Train σ", "Shift / σ"],
          result.numeric_drift.map((d) => [
            d.feature,
            d.train_mean,
            d.validation_mean,
            d.train_std,
            d.standardized_mean_shift === null
              ? d.constant_reference_changed
                ? "Constant baseline changed"
                : "Constant baseline unchanged"
              : d.standardized_mean_shift,
          ]),
        ),
      );
    } else
      $("results").append(
        el(
          "p",
          "No valid numeric feature was available for mean-drift calculation. Review the selected features and schema.",
          "hint",
        ),
      );
    if (result.distribution_checks_enabled) {
      const distributions = result.distribution_drift;
      $("results").append(
        el("h3", "Optional descriptive distribution distances"),
        bars(
          distributions
            .filter((d) => d.distance !== null)
            .map((d) => [d.feature, d.distance]),
          1,
        ),
        table(
          ["Feature", "Metric", "Distance (0–1)", "Threshold", "Status"],
          distributions.map((d) => [
            d.feature,
            d.metric,
            d.distance,
            d.threshold,
            d.status === "computed"
              ? d.flagged
                ? "Flagged"
                : "Within threshold"
              : "Skipped: invalid / missing values",
          ]),
        ),
        el(
          "p",
          "Flags use distance > threshold. These descriptive distances are not p-values, significance tests or a model-quality assessment.",
          "notice",
        ),
      );
    }
    $("results").append(el("p", result.limitations, "notice"));
    $("technical").textContent = JSON.stringify(result, null, 2);
    $("export-report").disabled = false;
    $("print").disabled = false;
  });
setupExports("mlops-doctor", () => report, config);

const evaluationPanel = el("section", "", "panel");
evaluationPanel.id = "prediction-evaluation";
evaluationPanel.append(
  el("h2", "Prediction evaluation and supported slices"),
  el(
    "p",
    "Import held-out actual and predicted values. Descriptive metrics are not a fairness or causal assessment. Classification CSV labels are text; regression columns must be numeric.",
  ),
);
for (const [id, label, type] of [
  ["eval-file", "Prediction CSV file", "file"],
  ["eval-csv", "Or paste predictions CSV", "textarea"],
  ["eval-minimum", "Minimum rows per group", "number"],
]) {
  const l = el("label", label),
    c = el(type === "textarea" ? "textarea" : "input");
  c.id = id;
  l.htmlFor = id;
  if (type !== "textarea") c.type = type;
  if (type === "file") c.accept = ".csv,text/csv";
  if (type === "textarea") c.maxLength = 2097152;
  if (type === "number") {
    c.min = 1;
    c.max = 10000;
    c.value = 5;
  }
  evaluationPanel.append(l, c);
}
for (const [id, label, values] of [
  ["eval-task", "Prediction task", ["classification", "regression"]],
  ["eval-actual", "Actual / true column", []],
  ["eval-predicted", "Prediction column", []],
  ["eval-group", "Optional group column", [""]],
]) {
  const f = selectField(id, label, values);
  evaluationPanel.append(f.box);
}
const evalPreview = el("button", "Preview and map prediction CSV"),
  evalExample = el("button", "Load synthetic prediction evaluation"),
  evalRun = el("button", "Evaluate predictions"),
  evalPreviewTable = el("div");
evalPreview.id = "eval-preview";
evalExample.id = "eval-example";
evalRun.id = "eval-run";
evaluationPanel.append(evalPreview, evalExample, evalRun, evalPreviewTable);
document.querySelector(".input-panel").append(evaluationPanel);
let evaluationSynthetic = false;
function evaluationDirty() {
  evaluationSynthetic = false;
  dirty();
}
evaluationPanel.addEventListener("input", (event) => {
  if (event?.target?.type !== "file") evaluationDirty();
});
evaluationPanel.addEventListener("change", (event) => {
  if (event?.target?.type !== "file") evaluationDirty();
});
async function previewEvaluation() {
  invalidate();
  const generation = getRevision(),
    r = await api("/csv/preview", { csv_text: $("eval-csv").value });
  assertCurrent(generation);
  replace(evalPreviewTable, previewRows(r.columns, r.preview));
  setOptions(
    $("eval-actual"),
    r.columns,
    r.columns.find((c) => /actual|true/i.test(c)) || r.columns[0],
  );
  setOptions(
    $("eval-predicted"),
    r.columns,
    r.columns.find((c) => /pred/i.test(c)) || r.columns[1],
  );
  setOptions(
    $("eval-group"),
    ["", ...r.columns],
    r.columns.find((c) => /group|slice/i.test(c)) || "",
  );
  return { generation };
}
evalPreview.onclick = async () => {
  try {
    evaluationDirty();
    await previewEvaluation();
    status(
      "Prediction CSV parsed. Choose task, mappings and minimum group support.",
    );
  } catch (e) {
    if (!isStale(e)) status(e.message, true);
  }
};
bindFile(
  "eval-file",
  "eval-csv",
  async () => {
    evaluationDirty();
    await previewEvaluation();
  },
  evaluationDirty,
);
evalExample.onclick = async () => {
  try {
    evaluationDirty();
    $("eval-csv").value =
      "actual,predicted,group\nyes,yes,A\nyes,no,A\nno,no,A\nno,yes,B\nyes,yes,B\nno,no,B";
    $("eval-task").value = "classification";
    $("eval-minimum").value = 2;
    const r = await previewEvaluation();
    assertCurrent(r.generation);
    evaluationSynthetic = true;
    $("data-origin").textContent =
      "Synthetic prediction evaluation · no real people";
    status(
      "Synthetic classification loaded: accuracy 4/6, two supported groups of three rows.",
    );
  } catch (e) {
    if (!isStale(e)) status(e.message, true);
  }
};
evalRun.onclick = () =>
  action(async () => {
    report = null;
    const input = {
        csv_text: $("eval-csv").value,
        task: $("eval-task").value,
        actual_column: $("eval-actual").value,
        predicted_column: $("eval-predicted").value,
        group_column: $("eval-group").value || null,
        minimum_support: Number($("eval-minimum").value),
      },
      generation = getRevision(),
      result = await api("/csv/evaluate", input);
    assertCurrent(generation);
    report = newReport(
      "MLOps Doctor prediction evaluation",
      input,
      result,
      evaluationSynthetic,
    );
    const summary = result.overall;
    replace(
      $("results"),
      el("h3", "Prediction metrics"),
      metrics(
        result.task === "classification"
          ? [
              ["Rows", summary.support],
              ["Accuracy", summary.accuracy],
              ["Macro F1", summary.macro.f1],
              ["Weighted F1", summary.weighted.f1],
            ]
          : [
              ["Rows", summary.support],
              ["MAE", summary.mae],
              ["RMSE", summary.rmse],
              ["R²", summary.r2 ?? "Undefined: constant actual"],
            ],
      ),
      el("p", result.method, "notice"),
    );
    if (result.task === "classification") {
      $("results").append(
        el("h3", "Confusion matrix · rows actual, columns predicted"),
        table(
          [
            "Actual / predicted",
            ...summary.labels.map((l) => `${l.type}: ${l.value}`),
          ],
          summary.confusion_matrix.map((row, i) => [
            `${summary.labels[i].type}: ${summary.labels[i].value}`,
            ...row,
          ]),
        ),
        table(
          ["Class", "Support", "Precision", "Recall", "F1"],
          summary.per_class.map((c) => [
            `${c.label_type}: ${c.label}`,
            c.support,
            c.precision,
            c.recall,
            c.f1,
          ]),
        ),
      );
    }
    $("results").append(
      el("h3", "Group summaries"),
      table(
        result.task === "classification"
          ? ["Group", "Rows", "Status", "Accuracy", "Macro F1", "Weighted F1"]
          : ["Group", "Rows", "Status", "MAE", "RMSE", "R²"],
        result.slices.map((g) =>
          [
            g.group,
            g.support,
            g.status,
            ...(result.task === "classification"
              ? [
                  g.metrics?.accuracy,
                  g.metrics?.macro.f1,
                  g.metrics?.weighted.f1,
                ]
              : [g.metrics?.mae, g.metrics?.rmse, g.metrics?.r2]),
          ].map((v) => v ?? "Not evaluated / undefined"),
        ),
      ),
    );
    const save = el("button", "Export prediction report"),
      configSave = el("button", "Export prediction configuration");
    save.onclick = () => download("prediction-report.json", report);
    configSave.onclick = () => download("prediction-config.json", input);
    $("results").append(save, configSave);
    $("technical").textContent = JSON.stringify(result, null, 2);
    $("export-report").disabled = false;
    $("print").disabled = false;
  });
