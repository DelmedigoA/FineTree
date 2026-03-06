from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import uvicorn

from .api import create_app
from .service import WorkspaceWebService


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FineTree web annotator.")
    parser.add_argument("input_path", nargs="?", default=None, help="Optional PDF file or page-images directory to open.")
    parser.add_argument("--images-dir", default=None, help="Optional page-images directory to import into the managed workspace.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload mode.")
    parser.add_argument("--data-root", default="data", help="Workspace data root.")
    parser.add_argument("--frontend-dist", default=None, help="Built frontend directory (default: frontend/dist).")
    parser.add_argument("--dpi", type=int, default=200, help="PDF extraction DPI when importing a PDF.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root).expanduser().resolve()
    frontend_dist = Path(args.frontend_dist).expanduser().resolve() if args.frontend_dist else None
    service = WorkspaceWebService(data_root)

    startup_doc_id: str | None = None
    if args.input_path or args.images_dir:
        raw_path = Path(args.images_dir or args.input_path).expanduser().resolve()
        if raw_path.is_file() and raw_path.suffix.lower() == ".pdf":
            document = service.import_pdf_path(raw_path, dpi=int(args.dpi))
            startup_doc_id = document.doc_id
        elif raw_path.is_dir():
            startup_doc_id = service.import_image_directory(raw_path).doc_id
        else:
            raise SystemExit(f"Unsupported startup path: {raw_path}")

    os.environ["FINETREE_WEB_DATA_ROOT"] = str(data_root)
    if frontend_dist is not None:
        os.environ["FINETREE_WEB_FRONTEND_DIST"] = str(frontend_dist)
    elif "FINETREE_WEB_FRONTEND_DIST" not in os.environ:
        default_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
        if default_dist.is_dir():
            os.environ["FINETREE_WEB_FRONTEND_DIST"] = str(default_dist)
    if startup_doc_id:
        os.environ["FINETREE_WEB_STARTUP_DOC_ID"] = startup_doc_id

    if args.reload:
        uvicorn.run(
            "finetree_annotator.web.api:create_app_from_env",
            factory=True,
            host=args.host,
            port=int(args.port),
            reload=True,
        )
        return 0

    app = create_app(
        data_root=data_root,
        startup_doc_id=startup_doc_id,
        frontend_dist=frontend_dist,
    )
    uvicorn.run(app, host=args.host, port=int(args.port), reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
