"""Local HTTP interface for CoSER summary generation and dataset export."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from config import ROOT, Settings
from summaries import ConnectionRequest, ResumeSummary, SourceKind, StartSummary, SummaryManager


def create_app(settings=None):
    settings = settings or Settings()
    summaries = SummaryManager(settings)

    @asynccontextmanager
    async def lifespan(app):
        summaries.recover()
        yield
        await summaries.shutdown()

    app = FastAPI(title="Hafvillai Dataset Studio", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.summaries = summaries
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_request_guard(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if (origin and origin != str(request.base_url).rstrip("/")) or request.headers.get("x-studio-request") != "1":
                return JSONResponse({"detail": "Use the local Dataset Studio page."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        if request.url.path == "/" or request.url.path.startswith(("/api/", "/static/")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "static/index.html")

    @app.get("/api/summaries/status")
    async def summary_status(source_kind: SourceKind | None = None):
        return summaries.status()

    @app.post("/api/summaries/check")
    async def summary_check(request: ConnectionRequest):
        return await summaries.check(request)

    @app.post("/api/summaries/start", status_code=202)
    async def summary_start(request: StartSummary):
        return await summaries.start(request)

    @app.post("/api/summaries/resume", status_code=202)
    async def summary_resume(request: ResumeSummary):
        return await summaries.resume(request)

    @app.post("/api/summaries/stop")
    async def summary_stop():
        return await summaries.stop()

    @app.get("/api/summaries/download")
    async def summary_download():
        path = summaries.artifact()
        return FileResponse(path, media_type="application/x-ndjson", filename=path.name)

    @app.post("/api/summaries/merge")
    async def summary_merge(source_kind: SourceKind = "coser"):
        return summaries.merge(source_kind)

    @app.get("/api/summaries/merged/{merge_id}")
    async def summary_merged_download(merge_id: str):
        path = summaries.merged_artifact(merge_id)
        return FileResponse(path, media_type="application/x-ndjson", filename=path.name)

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return app


app = create_app()
