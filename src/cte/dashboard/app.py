from pathlib import Path

from .data import load_dashboard_snapshot


def create_app(*, experiments_root: Path, parsed_docs_root: Path | None = None) -> object:
    try:
        from fastapi import FastAPI, HTTPException, Query, Request
        from fastapi.responses import FileResponse
        from fastapi.templating import Jinja2Templates
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Dashboard dependencies are missing. Install them with `uv sync`."
        ) from exc

    app = FastAPI(title="CTE Dashboard", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(Path(__file__).with_name("templates")))
    repo_root = Path.cwd().resolve()
    dashboard_root = Path(experiments_root)
    parsed_docs_root_path = (
        Path(parsed_docs_root) if parsed_docs_root is not None else Path("artifacts/parsed_docs")
    )

    @app.get("/")
    def dashboard_index(request: Request):
        snapshot = load_dashboard_snapshot(
            dashboard_root,
            parsed_docs_root=parsed_docs_root_path,
        )
        return templates.TemplateResponse(
            request,
            "dashboard_index.html",
            {
                "snapshot": snapshot,
                "refresh_seconds": 5,
            },
        )

    @app.get("/runs/{run_id}")
    def run_detail(request: Request, run_id: str):
        snapshot = load_dashboard_snapshot(
            dashboard_root,
            parsed_docs_root=parsed_docs_root_path,
        )
        run = snapshot.runs_by_id.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return templates.TemplateResponse(
            request,
            "dashboard_run_detail.html",
            {
                "snapshot": snapshot,
                "run": run,
                "refresh_seconds": 5,
            },
        )

    @app.get("/artifact")
    def artifact_file(path: str = Query(...)):
        try:
            file_path = _resolve_artifact_request_path(path, repo_root=repo_root)
        except RuntimeError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(file_path)

    return app


def run_dashboard(
    *,
    experiments_root: Path,
    parsed_docs_root: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Dashboard dependencies are missing. Install them with `uv sync`."
        ) from exc

    app = create_app(experiments_root=experiments_root, parsed_docs_root=parsed_docs_root)
    uvicorn.run(app, host=host, port=port)


def _resolve_artifact_request_path(path: str, *, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve(strict=False)
    else:
        candidate = candidate.resolve(strict=False)

    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError("artifact path is outside repository root") from exc

    return candidate
