from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from .submission import SUBMISSION_FIELD_NAMES


def israel_now_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Jerusalem")).isoformat()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_logging_rows(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object on line {line_number} of {path}")
        rows.append(payload)
    return rows


def _find_package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _load_torch() -> Any | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def collect_environment_snapshot(*, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    active_env = dict(env or os.environ)
    torch = _load_torch()
    gpu_names: list[str] = []
    gpu_count = 0
    cuda_runtime_version: str | None = None
    if torch is not None:
        try:
            gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            gpu_names = [str(torch.cuda.get_device_name(index)) for index in range(gpu_count)]
            cuda_runtime_version = str(torch.version.cuda) if getattr(torch, "version", None) else None
        except Exception:
            gpu_count = 0
            gpu_names = []
    nvidia_smi_output: str | None = None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            nvidia_smi_output = completed.stdout.strip() or None
            if not gpu_names and nvidia_smi_output:
                gpu_names = [line.split(",", 1)[0].strip() for line in nvidia_smi_output.splitlines() if line.strip()]
                gpu_count = len(gpu_names)
    except Exception:
        nvidia_smi_output = None
    return {
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "hostname": socket.gethostname(),
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
        "cuda_runtime_version": cuda_runtime_version,
        "ms_swift_version": _find_package_version("ms-swift") or _find_package_version("swift"),
        "transformers_version": _find_package_version("transformers"),
        "cuda_visible_devices": active_env.get("CUDA_VISIBLE_DEVICES"),
        "max_pixels_env": active_env.get("MAX_PIXELS"),
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "gpu_used": " | ".join(gpu_names) if gpu_names else None,
        "nvidia_smi": nvidia_smi_output,
    }


def _parse_step_progress(row: Mapping[str, Any]) -> tuple[int | None, int | None]:
    text = str(row.get("global_step/max_steps") or "").strip()
    if "/" not in text:
        return None, None
    left, right = text.split("/", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None, None


def _checkpoint_step_map(output_dir: Path) -> dict[int, Path]:
    step_map: dict[int, Path] = {}
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match is None:
            continue
        step_map[int(match.group(1))] = path
    return step_map


def select_best_checkpoint(output_dir: Path | str, logging_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    step_map = _checkpoint_step_map(output_path)
    eval_candidates: list[dict[str, Any]] = []
    for row in logging_rows:
        if "eval_loss" not in row:
            continue
        try:
            eval_loss = float(row["eval_loss"])
        except Exception:
            continue
        global_step, max_steps = _parse_step_progress(row)
        epoch = row.get("epoch")
        epoch_value = float(epoch) if isinstance(epoch, (int, float)) else None
        eval_token_acc = row.get("eval_token_acc")
        if global_step is None or global_step not in step_map:
            continue
        eval_candidates.append(
            {
                "checkpoint_name": step_map[global_step].name,
                "checkpoint_path": str(step_map[global_step]),
                "epoch": epoch_value,
                "global_step": global_step,
                "max_steps": max_steps,
                "eval_loss": eval_loss,
                "eval_token_acc": float(eval_token_acc) if isinstance(eval_token_acc, (int, float)) else None,
                "selection_metric": "eval_loss",
                "selection_reason": "lowest_eval_loss",
            }
        )
    if eval_candidates:
        best = sorted(
            eval_candidates,
            key=lambda item: (
                float(item["eval_loss"]),
                -(item["epoch"] if item["epoch"] is not None else -1.0),
                -int(item["global_step"]),
            ),
        )[0]
        best["adapter_only"] = True
        return best
    if step_map:
        final_step = max(step_map)
        return {
            "checkpoint_name": step_map[final_step].name,
            "checkpoint_path": str(step_map[final_step]),
            "epoch": None,
            "global_step": final_step,
            "max_steps": None,
            "eval_loss": None,
            "eval_token_acc": None,
            "selection_metric": "fallback",
            "selection_reason": "final_checkpoint",
            "adapter_only": True,
        }
    raise FileNotFoundError(f"No checkpoint-* directories found under {output_path}")


def _coerce_json_object(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    raise ValueError("Prediction payload must be a JSON object.")


def _maybe_parse_json_string(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _maybe_extract_json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    parsed = _maybe_parse_json_string(text)
    if parsed is not None:
        return parsed
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def extract_prediction_payload(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        if isinstance(row.get("pages"), list) or isinstance(row.get("facts"), list):
            return _coerce_json_object(row)
        for key in ("prediction_json", "parsed_prediction", "prediction", "response", "generated_text", "output", "text"):
            value = row.get(key)
            if isinstance(value, dict):
                if isinstance(value.get("pages"), list) or isinstance(value.get("facts"), list):
                    return value
            parsed = _maybe_extract_json_object(value)
            if parsed is not None:
                return parsed
        messages = row.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if not isinstance(message, dict):
                    continue
                parsed = _maybe_extract_json_object(message.get("content"))
                if parsed is not None:
                    return parsed
    if isinstance(row, str):
        parsed = _maybe_extract_json_object(row)
        if parsed is not None:
            return parsed
    raise ValueError(
        "Could not extract a benchmark JSON payload from inference output row. "
        "Refusing to fall back to labels or other non-prediction fields."
    )


def materialize_prediction_json_files(
    result_jsonl_path: Path | str,
    output_dir: Path | str,
    *,
    prefix: str = "pred_",
) -> list[Path]:
    result_path = Path(result_jsonl_path).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for index, line in enumerate(result_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        prediction = extract_prediction_payload(payload)
        path = target_dir / f"{prefix}{index:04d}.json"
        path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written_paths.append(path)
    if not written_paths:
        raise ValueError(f"No prediction rows were written from {result_path}")
    return written_paths


def build_benchmark_model_metadata(
    *,
    training_args: Mapping[str, Any],
    environment: Mapping[str, Any],
    checkpoint_name: str,
) -> dict[str, Any]:
    metadata_out: dict[str, Any] = {field_name: None for field_name in SUBMISSION_FIELD_NAMES}
    metadata_out.update({str(key): value for key, value in training_args.items() if str(key) in metadata_out})
    metadata_out["checkpoint_name"] = checkpoint_name
    metadata_out["validation_dataset"] = training_args.get("validation_dataset") or training_args.get("val_dataset")
    metadata_out["CUDA_VISIBLE_DEVICES"] = (
        training_args.get("CUDA_VISIBLE_DEVICES")
        if training_args.get("CUDA_VISIBLE_DEVICES") is not None
        else environment.get("cuda_visible_devices")
    )
    metadata_out["MAX_PIXELS"] = training_args.get("MAX_PIXELS") or environment.get("max_pixels_env")
    metadata_out["gpu_used"] = environment.get("gpu_used")
    metadata_out["torch_env_used"] = " | ".join(
        str(value)
        for value in (
            f"torch {environment.get('torch_version')}" if environment.get("torch_version") else None,
            f"cuda {environment.get('cuda_runtime_version')}" if environment.get("cuda_runtime_version") else None,
            f"python {environment.get('python_version')}" if environment.get("python_version") else None,
        )
        if value
    ) or None
    metadata_out["platform"] = environment.get("platform")
    return metadata_out


def build_submission_info_payload(
    *,
    model_metadata: Mapping[str, Any],
    training_args: Mapping[str, Any],
    environment: Mapping[str, Any],
    run: Mapping[str, Any],
    selected_checkpoint: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_metadata": {str(key): value for key, value in model_metadata.items()},
        "training_args": {str(key): value for key, value in training_args.items()},
        "environment": {str(key): value for key, value in environment.items()},
        "run": {str(key): value for key, value in run.items()},
        "selected_checkpoint": {str(key): value for key, value in selected_checkpoint.items()},
        "artifacts": {str(key): value for key, value in artifacts.items()},
    }


def write_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination
