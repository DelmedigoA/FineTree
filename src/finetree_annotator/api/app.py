"""FastAPI application factory for the FineTree web API."""
from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .deps import set_data_root
from .workspace_router import router as workspace_router
from .annotation_router import router as annotation_router
from .image_router import router as image_router
from .schema_router import router as schema_router
from .ai_router import router as ai_router
from .batch_infer_router import router as batch_infer_router


def create_app(*, data_root: Path | None = None) -> FastAPI:
    if data_root is not None:
        set_data_root(data_root)

    app = FastAPI(
        title="FineTree Annotator API",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(workspace_router)
    app.include_router(annotation_router)
    app.include_router(image_router)
    app.include_router(schema_router)
    app.include_router(ai_router)
    app.include_router(batch_infer_router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="FineTree Web API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-root", type=str, default="data")
    args = parser.parse_args()

    import uvicorn

    set_data_root(Path(args.data_root))
    app = create_app(data_root=Path(args.data_root))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
