from __future__ import annotations

import argparse
import glob
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .config import FinetuneConfig, load_finetune_config, resolve_hf_token

_DATA_SYNC_HINT = (
    "If this is a fresh clone, sync dataset folders into the repo before training: "
    "data/annotations/ and data/pdf_images/."
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str


def _iter_annotation_files(pattern: str) -> Iterable[Path]:
    try:
        candidates = [Path(p) for p in Path().glob(pattern)]
    except (ValueError, NotImplementedError):
        candidates = [Path(p) for p in glob.glob(pattern)]
    for p in sorted(candidates):
        if p.is_file():
            yield p


def _resolve_page_image_path(cfg: FinetuneConfig, payload: dict, image_name: str) -> Path:
    images_dir_raw = str(payload.get("images_dir") or "").strip()
    images_dir = Path(images_dir_raw)
    if not images_dir.is_absolute():
        images_dir = (cfg.data.images_root / images_dir).resolve()
    return (images_dir / image_name).resolve()


def _check_training_stack() -> CheckResult:
    required = ("torch", "transformers", "datasets", "trl", "peft", "huggingface_hub", "unsloth", "yaml")
    missing: List[str] = []
    for mod in required:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        return CheckResult(
            name="python-stack",
            ok=False,
            message=f"Missing modules: {', '.join(missing)}",
        )
    return CheckResult(name="python-stack", ok=True, message="All required modules are importable.")


def _check_cuda() -> CheckResult:
    try:
        import torch
    except Exception as exc:
        return CheckResult(name="cuda", ok=False, message=f"PyTorch import failed: {exc}")

    if not torch.cuda.is_available():
        return CheckResult(name="cuda", ok=False, message="torch.cuda.is_available() is False")

    try:
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if count > 0 else "unknown"
        _ = torch.randn((1024, 1024), device="cuda") @ torch.randn((1024, 1024), device="cuda")
        return CheckResult(name="cuda", ok=True, message=f"CUDA OK on {count} GPU(s). First GPU: {name}")
    except Exception as exc:
        return CheckResult(name="cuda", ok=False, message=f"CUDA smoke op failed: {exc}")


def _check_config_paths(cfg: FinetuneConfig) -> CheckResult:
    if cfg.prompt.use_custom_prompt and not cfg.prompt.prompt_path.is_file():
        return CheckResult(
            name="config-paths",
            ok=False,
            message=f"Prompt file not found: {cfg.prompt.prompt_path}",
        )

    ann_files = list(_iter_annotation_files(cfg.data.annotations_glob))
    if not ann_files:
        return CheckResult(
            name="config-paths",
            ok=False,
            message=f"No annotation files matched: {cfg.data.annotations_glob}. {_DATA_SYNC_HINT}",
        )
    return CheckResult(name="config-paths", ok=True, message=f"Found {len(ann_files)} annotation file(s).")


def _check_annotations_and_images(cfg: FinetuneConfig, probe_pages: int = 200) -> CheckResult:
    ann_files = list(_iter_annotation_files(cfg.data.annotations_glob))
    if not ann_files:
        return CheckResult(name="annotations", ok=False, message=f"No annotations to inspect. {_DATA_SYNC_HINT}")

    pages_seen = 0
    fact_pages = 0
    missing_images: List[str] = []
    invalid_json: List[str] = []

    for ann in ann_files:
        try:
            payload = json.loads(ann.read_text(encoding="utf-8"))
        except Exception:
            invalid_json.append(str(ann))
            continue
        pages = payload.get("pages") if isinstance(payload.get("pages"), list) else []
        for page in pages:
            if not isinstance(page, dict):
                continue
            pages_seen += 1
            facts = page.get("facts") if isinstance(page.get("facts"), list) else []
            if facts:
                fact_pages += 1
            image_name = str(page.get("image") or "").strip()
            if image_name:
                p = _resolve_page_image_path(cfg, payload, image_name)
                if not p.is_file():
                    missing_images.append(str(p))
            if pages_seen >= probe_pages:
                break
        if pages_seen >= probe_pages:
            break

    if invalid_json:
        return CheckResult(name="annotations", ok=False, message=f"Invalid JSON in: {invalid_json[0]}")
    if pages_seen == 0:
        return CheckResult(name="annotations", ok=False, message="No pages found in annotations.")
    if not cfg.data.include_empty_pages and fact_pages == 0:
        return CheckResult(name="annotations", ok=False, message="No pages with facts found; dataset would be empty.")
    if missing_images:
        preview = missing_images[:3]
        tail = f" (+{len(missing_images)-3} more)" if len(missing_images) > 3 else ""
        return CheckResult(name="annotations", ok=False, message=f"Missing image files: {preview}{tail}")

    return CheckResult(
        name="annotations",
        ok=True,
        message=f"Scanned {pages_seen} page(s), {fact_pages} page(s) with facts, all images resolved.",
    )


def _check_hf_connectivity(cfg: FinetuneConfig, force: bool = False) -> CheckResult:
    if not force and not cfg.push_to_hub.enabled:
        return CheckResult(
            name="huggingface",
            ok=True,
            message="Push disabled in config; HF connectivity check skipped.",
        )

    token = resolve_hf_token(cfg.push_to_hub)
    if not token:
        return CheckResult(
            name="huggingface",
            ok=False,
            message=(
                "Missing Hugging Face token. Set push_to_hub.hf_token or export one of: "
                "FINETREE_HF_TOKEN, HF_TOKEN, HUGGINGFACE_HUB_TOKEN."
            ),
        )
    if not str(token).startswith("hf_"):
        return CheckResult(
            name="huggingface",
            ok=False,
            message="push_to_hub.hf_token does not look like a Hugging Face token (expected prefix hf_).",
        )

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return CheckResult(name="huggingface", ok=False, message=f"huggingface_hub import failed: {exc}")

    try:
        api = HfApi(token=token)
        identity = api.whoami()
    except Exception as exc:
        return CheckResult(name="huggingface", ok=False, message=f"HF auth/network check failed: {exc}")

    user_name = identity.get("name") if isinstance(identity, dict) else None
    if cfg.push_to_hub.enabled and cfg.push_to_hub.repo_id:
        try:
            api.create_repo(repo_id=str(cfg.push_to_hub.repo_id), private=True, exist_ok=True)
        except Exception as exc:
            return CheckResult(
                name="huggingface",
                ok=False,
                message=f"HF repo access/create check failed for {cfg.push_to_hub.repo_id}: {exc}",
            )
    return CheckResult(
        name="huggingface",
        ok=True,
        message=f"HF auth/network OK for account: {user_name or 'unknown'}.",
    )


def run_preflight(
    cfg: FinetuneConfig,
    *,
    check_stack: bool = True,
    check_cuda: bool = True,
    check_data: bool = True,
    check_hf: bool = False,
    probe_pages: int = 200,
) -> List[CheckResult]:
    checks: List[CheckResult] = []
    if check_stack:
        checks.append(_check_training_stack())
    if check_cuda:
        checks.append(_check_cuda())
    checks.append(_check_config_paths(cfg))
    if check_data:
        checks.append(_check_annotations_and_images(cfg, probe_pages=probe_pages))
    if check_hf:
        checks.append(_check_hf_connectivity(cfg, force=True))
    elif cfg.push_to_hub.enabled:
        checks.append(_check_hf_connectivity(cfg, force=False))
    return checks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preflight checks before GPU fine-tuning.")
    parser.add_argument("--config", required=True, help="Path to fine-tune YAML config.")
    parser.add_argument("--skip-stack", action="store_true", help="Skip Python package import checks.")
    parser.add_argument("--skip-cuda", action="store_true", help="Skip CUDA smoke checks.")
    parser.add_argument("--skip-data", action="store_true", help="Skip annotation/image integrity checks.")
    parser.add_argument("--check-hf", action="store_true", help="Force Hugging Face network/auth check.")
    parser.add_argument("--probe-pages", type=int, default=200, help="Max pages to scan during data preflight.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_finetune_config(args.config)
    checks = run_preflight(
        cfg,
        check_stack=not args.skip_stack,
        check_cuda=not args.skip_cuda,
        check_data=not args.skip_data,
        check_hf=bool(args.check_hf),
        probe_pages=max(1, int(args.probe_pages)),
    )

    failed = False
    for check in checks:
        prefix = "PASS" if check.ok else "FAIL"
        print(f"[{prefix}] {check.name}: {check.message}")
        if not check.ok:
            failed = True

    if failed:
        print("Preflight failed. Fix issues before starting fine-tune.")
        return 2
    print("Preflight passed.")
    return 0


__all__ = ["CheckResult", "run_preflight", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
