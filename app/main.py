from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .core import Request, analyze
from .guard import InputGuard

app = FastAPI(title="MLOps Doctor", version="0.2.0")
app.add_middleware(InputGuard)
STATIC = Path(__file__).with_name("static")
app.mount("/assets", StaticFiles(directory=STATIC), name="assets")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/analyze")
def run(request: Request):
    try:
        return analyze(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


from .csv_import import CSVPreview, parse_csv, convert_rows
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


@app.post("/csv/preview")
def preview_csv(request: CSVPreview):
    try:
        headers, rows = parse_csv(request.csv_text)
        return {"columns": headers, "row_count": len(rows), "preview": rows[:8]}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class CSVDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train_csv: str = Field(min_length=1, max_length=2097152)
    validation_csv: str = Field(min_length=1, max_length=2097152)
    columns: dict[str, Literal["string", "number", "boolean"]] = Field(alias="schema")
    id_column: str
    target: str
    features: list[str]
    drift_threshold: float = Field(default=1, gt=0, allow_inf_nan=False)


@app.post("/csv/analyze")
def diagnose_csv(request: CSVDiagnosis):
    try:
        _, train = parse_csv(request.train_csv)
        _, validation = parse_csv(request.validation_csv)
        data = Request(
            train=convert_rows(train, request.columns),
            validation=convert_rows(validation, request.columns),
            schema=request.columns,
            id_column=request.id_column,
            target=request.target,
            features=request.features,
            drift_threshold=request.drift_threshold,
        )
        return analyze(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/guide")
def user_guide():
    return FileResponse(STATIC / "guide.html")
