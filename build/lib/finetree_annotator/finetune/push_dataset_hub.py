from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import HfApi


def build_dataset(config_path: Path) -> None:
    from .dataset_builder import main as build_main

    build_main(["--config", str(config_path)])


def _rows_from_chat_jsonl(split_path: Path) -> list[dict[str, str]]:
    if not split_path.is_file():
        return []

    rows: list[dict[str, str]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        messages = sample.get("messages") if isinstance(sample.get("messages"), list) else []
        if len(messages) < 2:
            continue

        user_content = messages[0].get("content") if isinstance(messages[0], dict) else None
        assistant_content = messages[1].get("content") if isinstance(messages[1], dict) else None
        if not isinstance(user_content, list) or not isinstance(assistant_content, list):
            continue

        image_path = ""
        instruction = ""
        text = ""

        for part in user_content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image" and isinstance(part.get("image"), str):
                image_path = part["image"]
            elif part.get("type") == "text" and isinstance(part.get("text"), str):
                instruction = part["text"]

        if assistant_content and isinstance(assistant_content[0], dict):
            text_val = assistant_content[0].get("text")
            if isinstance(text_val, str):
                text = text_val

        if image_path and instruction and text:
            rows.append(
                {
                    "image": image_path,
                    "instruction": instruction,
                    "text": text,
                }
            )
    return rows


def build_hf_dataset(root: Path) -> tuple[DatasetDict, int, int]:
    train_in = root / "data/finetune/train.jsonl"
    val_in = root / "data/finetune/val.jsonl"
    train_rows = _rows_from_chat_jsonl(train_in)
    val_rows = _rows_from_chat_jsonl(val_in)

    features = Features(
        {
            "image": Image(),
            "instruction": Value("string"),
            "text": Value("string"),
        }
    )
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_rows, features=features),
            "validation": Dataset.from_list(val_rows, features=features),
        }
    )
    return dataset, len(train_rows), len(val_rows)


def export_for_hf(root: Path, export_dir: Path) -> Tuple[int, int]:
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
        for row in _rows_from_chat_jsonl(src_path):
            src_img = Path(row["image"])
            if not src_img.is_absolute():
                src_img = (root / src_img).resolve()
            if not src_img.exists():
                raise FileNotFoundError(f"Missing source image: {src_img}")

            doc_id = src_img.parent.name or "unknown_doc"
            rel_img = Path("images") / doc_id / src_img.name
            dst_img = export_dir / rel_img
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            out_lines.append(
                json.dumps(
                    {
                        "image": rel_img.as_posix(),
                        "instruction": row["instruction"].replace(str(src_img), rel_img.as_posix()),
                        "text": row["text"],
                    },
                    ensure_ascii=False,
                )
            )

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


def _default_repo_id(api: HfApi) -> str:
    username = api.whoami().get("name") or "user"
    return f"{username}/FineTree-annotated-pages"


def _resolve_doppler_scope_args() -> list[str]:
    args: list[str] = []
    project = (os.getenv("DOPPLER_PROJECT") or "").strip()
    config = (os.getenv("DOPPLER_CONFIG") or "").strip()
    if project:
        args.extend(["--project", project])
    if config:
        args.extend(["--config", config])
    return args


def _token_from_doppler() -> Optional[str]:
    if shutil.which("doppler") is None:
        return None
    scope_args = _resolve_doppler_scope_args()
    for secret in ("HF_TOKEN", "FINETREE_HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        cmd = ["doppler", "secrets", "get", secret, "--plain", *scope_args]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5)
        except Exception:
            continue
        token = proc.stdout.strip()
        if token:
            return token
    return None


def resolve_hf_token(explicit_token: Optional[str]) -> Optional[str]:
    return (
        (explicit_token or "").strip()
        or (os.getenv("FINETREE_HF_TOKEN") or "").strip()
        or (os.getenv("HF_TOKEN") or "").strip()
        or (os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
        or (os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()
        or _token_from_doppler()
    )


def push_to_hf(dataset: DatasetDict, token: str, repo_id: str | None, private: bool = True) -> str:
    api = HfApi(token=token)
    if repo_id is None:
        repo_id = _default_repo_id(api)

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    dataset.push_to_hub(repo_id=repo_id, token=token, private=private)
    return repo_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and push FineTree dataset with columns: image, instruction, text.")
    parser.add_argument("--config", default="configs/finetune_qwen35a3_vl.yaml")
    parser.add_argument("--repo-id", default=None, help="HF dataset repo id, e.g. username/FineTree-annotated-pages")
    parser.add_argument("--token", default=None, help="HF token (or use FINETREE_HF_TOKEN/HF_TOKEN/Doppler)")
    parser.add_argument("--export-dir", default="artifacts/hf_dataset_export")
    parser.add_argument("--public", action="store_true", help="Create/push dataset as public.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(".").resolve()
    token = resolve_hf_token(args.token)
    if not token:
        raise RuntimeError(
            "Missing HF token. Pass --token, export FINETREE_HF_TOKEN/HF_TOKEN, or configure HF_TOKEN in Doppler."
        )

    config_path = (root / args.config).resolve()
    export_dir = (root / args.export_dir).resolve()

    build_dataset(config_path)
    dataset, train_rows, val_rows = build_hf_dataset(root)
    repo = push_to_hf(dataset, token=token, repo_id=args.repo_id, private=not args.public)

    train_rows, val_rows = export_for_hf(root, export_dir)

    print(f"PUSHED: {repo}")
    print(f"TRAIN_ROWS: {train_rows}")
    print(f"VAL_ROWS: {val_rows}")
    print(f"EXPORT_DIR: {export_dir}")
    return 0


__all__ = ["build_dataset", "build_hf_dataset", "export_for_hf", "push_to_hf", "resolve_hf_token", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
