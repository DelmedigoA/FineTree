from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    ApiAppConfig,
    ApiApplyEntityRequest,
    ApiBooleanToggleRequest,
    ApiDocumentDetail,
    ApiDocumentSaveRequest,
    ApiDocumentSaveResponse,
    ApiDocumentValidateResponse,
    ApiExtractionRequest,
    ApiExtractionResponse,
    ApiImportJsonRequest,
    ApiSchemaResponse,
    ApiUploadPdfRequest,
    ApiWorkspaceDocumentSummary,
)
from .service import WorkspaceWebService


def _frontend_dist_from_env() -> Path | None:
    raw = str(os.getenv("FINETREE_WEB_FRONTEND_DIST") or "").strip()
    if not raw:
        candidate = Path(__file__).resolve().parents[3] / "frontend" / "dist"
        return candidate if candidate.is_dir() else None
    path = Path(raw).expanduser().resolve()
    return path if path.is_dir() else None


def _data_root_from_env() -> Path:
    raw = str(os.getenv("FINETREE_WEB_DATA_ROOT") or "").strip()
    if not raw:
        return Path("data")
    return Path(raw).expanduser().resolve()


def create_app(
    *,
    data_root: Path | None = None,
    startup_doc_id: str | None = None,
    frontend_dist: Path | None = None,
) -> FastAPI:
    resolved_data_root = Path(data_root) if data_root is not None else _data_root_from_env()
    resolved_frontend_dist = frontend_dist if frontend_dist is not None else _frontend_dist_from_env()
    app = FastAPI(title="FineTree Annotator Web API", version="0.1.0")
    service = WorkspaceWebService(resolved_data_root)
    app.state.workspace_service = service
    app.state.startup_doc_id = startup_doc_id
    app.state.frontend_dist = resolved_frontend_dist

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:4173",
            "http://localhost:4173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/app-config", response_model=ApiAppConfig)
    def app_config() -> ApiAppConfig:
        return service.app_config(
            startup_doc_id=app.state.startup_doc_id,
            frontend_dist=app.state.frontend_dist,
        )

    @app.get("/api/schema", response_model=ApiSchemaResponse)
    def schema() -> ApiSchemaResponse:
        return ApiSchemaResponse.build()

    @app.get("/api/workspace/documents", response_model=list[ApiWorkspaceDocumentSummary])
    def list_documents() -> list[ApiWorkspaceDocumentSummary]:
        return service.list_documents()

    @app.post("/api/workspace/import-pdf", response_model=ApiDocumentDetail)
    def import_pdf(request: ApiUploadPdfRequest) -> ApiDocumentDetail:
        return service.import_pdf_upload(
            filename=request.filename,
            content_b64=request.content_b64,
            dpi=request.dpi,
        )

    @app.post("/api/workspace/documents/{doc_id}/prepare", response_model=ApiDocumentDetail)
    def prepare_document(doc_id: str, dpi: int = 200) -> ApiDocumentDetail:
        return service.prepare_document(doc_id, dpi=dpi)

    @app.post("/api/workspace/documents/{doc_id}/checked", response_model=ApiWorkspaceDocumentSummary)
    def set_checked(doc_id: str, request: ApiBooleanToggleRequest) -> ApiWorkspaceDocumentSummary:
        return service.set_checked(doc_id, request.value)

    @app.post("/api/workspace/documents/{doc_id}/reviewed", response_model=ApiWorkspaceDocumentSummary)
    def set_reviewed(doc_id: str, request: ApiBooleanToggleRequest) -> ApiWorkspaceDocumentSummary:
        return service.set_reviewed(doc_id, request.value)

    @app.get("/api/documents/{doc_id}", response_model=ApiDocumentDetail)
    def get_document(doc_id: str) -> ApiDocumentDetail:
        return service.get_document(doc_id)

    @app.get("/api/documents/{doc_id}/pages/{page_image}/image")
    def get_page_image(doc_id: str, page_image: str) -> FileResponse:
        return FileResponse(service.page_image_path(doc_id, page_image))

    @app.put("/api/documents/{doc_id}", response_model=ApiDocumentSaveResponse)
    def save_document(doc_id: str, request: ApiDocumentSaveRequest) -> ApiDocumentSaveResponse:
        return service.save_document(doc_id, request)

    @app.post("/api/documents/{doc_id}/validate", response_model=ApiDocumentValidateResponse)
    def validate_document(doc_id: str, request: ApiDocumentSaveRequest) -> ApiDocumentValidateResponse:
        return service.validate_document(doc_id, request)

    @app.post("/api/documents/{doc_id}/import-json", response_model=ApiDocumentDetail)
    def import_json(doc_id: str, request: ApiImportJsonRequest) -> ApiDocumentDetail:
        return service.import_json(
            doc_id,
            request.payload,
            default_page_image=request.default_page_image,
            normalized_1000=bool(request.normalized_1000),
        )

    @app.post("/api/documents/{doc_id}/apply-entity", response_model=ApiDocumentDetail)
    def apply_entity(doc_id: str, request: ApiApplyEntityRequest) -> ApiDocumentDetail:
        return service.apply_entity_name(
            doc_id,
            entity_name=request.entity_name,
            overwrite_existing=bool(request.overwrite_existing),
        )

    @app.post("/api/documents/{doc_id}/pages/{page_image}/extract", response_model=ApiExtractionResponse)
    def extract_page(doc_id: str, page_image: str, request: ApiExtractionRequest) -> ApiExtractionResponse:
        return service.extract_page(doc_id, page_image, request)

    if resolved_frontend_dist is not None and resolved_frontend_dist.is_dir():
        assets_dir = resolved_frontend_dist / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

        @app.get("/", include_in_schema=False)
        def spa_index() -> FileResponse:
            return FileResponse(resolved_frontend_dist / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa_routes(full_path: str):
            if full_path.startswith("api/"):
                return HTMLResponse(status_code=404, content="Not found")
            candidate = resolved_frontend_dist / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(resolved_frontend_dist / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def dev_index() -> HTMLResponse:
            return HTMLResponse(
                """
                <!doctype html>
                <html lang="en">
                  <head>
                    <meta charset="utf-8" />
                    <title>FineTree Annotator Web</title>
                    <style>
                      body { font-family: "Avenir Next", "Helvetica Neue", sans-serif; margin: 0; padding: 32px; background: #0f172a; color: #e2e8f0; }
                      code { background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 6px; }
                      a { color: #7dd3fc; }
                    </style>
                  </head>
                  <body>
                    <h1>FineTree Annotator Web API</h1>
                    <p>The backend is running, but no built React app was found.</p>
                    <p>Run <code>cd frontend && npm install && npm run dev</code> for local development, or <code>npm run build</code> to let FastAPI serve the static app.</p>
                  </body>
                </html>
                """
            )

    return app


def create_app_from_env() -> FastAPI:
    startup_doc_id = str(os.getenv("FINETREE_WEB_STARTUP_DOC_ID") or "").strip() or None
    return create_app(
        data_root=_data_root_from_env(),
        startup_doc_id=startup_doc_id,
        frontend_dist=_frontend_dist_from_env(),
    )
