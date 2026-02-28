#!/usr/bin/env python3
# Save as push_finetree_dataset_to_hf.py and run:
#   python push_finetree_dataset_to_hf.py --token hf_xxx
# or:
#   FINETREE_HF_TOKEN=hf_xxx python push_finetree_dataset_to_hf.py

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import BadRequestError, HfHubHTTPError


def build_dataset(config_path: Path) -> None:
    from finetree_annotator.finetune.dataset_builder import main as build_main

    build_main(["--config", str(config_path)])


def export_for_hf(root: Path, export_dir: Path) -> tuple[int, int]:
    train_in = root / "data/finetune/train.jsonl"
    val_in = root / "data/finetune/val.jsonl"

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    def export_split(src_path: Path, dst_name: str) -> int:
        dst_path = export_dir / dst_name
        if not src_path.exists():
            dst_path.write_text("", encoding="utf-8")
            return 0

        out_lines: list[str] = []
        for line in src_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            meta = sample.get("metadata") or {}
            doc_id = str(meta.get("document_id") or "unknown_doc")

            user_content = sample["messages"][0]["content"]
            img_block = user_content[0]
            src_img = Path(img_block["image"])
            if not src_img.is_absolute():
                src_img = (root / src_img).resolve()
            if not src_img.exists():
                raise FileNotFoundError(f"Missing source image: {src_img}")

            rel_img = Path("images") / doc_id / src_img.name
            dst_img = export_dir / rel_img
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            img_block["image"] = rel_img.as_posix()

            text_block = user_content[1]
            if isinstance(text_block, dict) and isinstance(text_block.get("text"), str):
                text_block["text"] = text_block["text"].replace(str(src_img), rel_img.as_posix())

            out_lines.append(json.dumps(sample, ensure_ascii=False))

        dst_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        return len(out_lines)

    train_rows = export_split(train_in, "train.jsonl")
    val_rows = export_split(val_in, "val.jsonl")

    (export_dir / "README.md").write_text(
        "# FineTree Annotated Dataset\n\n"
        "Generated from current repository annotations.\n\n"
        f"- Train rows: {train_rows}\n"
        f"- Val rows: {val_rows}\n",
        encoding="utf-8",
    )
    return train_rows, val_rows


def push_to_hf(export_dir: Path, token: str, repo_id: str | None) -> str:
    api = HfApi(token=token)
    if repo_id is None:
        username = api.whoami().get("name") or "user"
        repo_id = f"{username}/FineTree-annotated-pages"

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    try:
        # Remove remote LFS rules if they exist to avoid png->LFS pointer failures.
        api.delete_file(
            path_in_repo=".gitattributes",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Remove .gitattributes to avoid LFS pointer issues",
        )
    except HfHubHTTPError:
        pass

    try:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(export_dir),
            commit_message="Upload FineTree annotated dataset (jsonl + images)",
        )
        return repo_id
    except BadRequestError as exc:
        msg = str(exc)
        if "LFS pointer pointed to a file that does not exist" not in msg:
            raise

        # Fallback to a fresh repo id with no historical LFS attributes.
        fallback_repo_id = f"{repo_id}-nolfs-{int(time.time())}"
        print(
            "Detected HF LFS pointer rejection on target repo. "
            f"Retrying with fresh repo: {fallback_repo_id}"
        )
        api.create_repo(repo_id=fallback_repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=fallback_repo_id,
            repo_type="dataset",
            folder_path=str(export_dir),
            commit_message="Upload FineTree annotated dataset (jsonl + images)",
        )
        return fallback_repo_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_qwen35a3_vl.yaml")
    parser.add_argument("--repo-id", default=None, help="HF dataset repo id, e.g. username/FineTree-annotated-pages")
    parser.add_argument("--token", default=None, help="HF token (or use FINETREE_HF_TOKEN env var)")
    parser.add_argument("--export-dir", default="artifacts/hf_dataset_export")
    args = parser.parse_args()

    root = Path(".").resolve()
    token = args.token or __import__("os").environ.get("FINETREE_HF_TOKEN")
    if not token:
        raise RuntimeError("Missing HF token. Pass --token or set FINETREE_HF_TOKEN.")

    config_path = (root / args.config).resolve()
    export_dir = (root / args.export_dir).resolve()

    build_dataset(config_path)
    train_rows, val_rows = export_for_hf(root, export_dir)
    repo = push_to_hf(export_dir, token=token, repo_id=args.repo_id)

    print(f"PUSHED: {repo}")
    print(f"TRAIN_ROWS: {train_rows}")
    print(f"VAL_ROWS: {val_rows}")
    print(f"EXPORT_DIR: {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
