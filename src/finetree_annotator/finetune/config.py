from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator


class RunConfig(BaseModel):
    name: str = "qwen35a3-vl-ft"
    seed: int = 3407
    output_dir: Path = Path("artifacts/qwen35a3-vl-ft")


class ModelConfig(BaseModel):
    base_model: str = "unsloth/Qwen3-VL-32B-Instruct"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "bfloat16"
    trust_remote_code: bool = True


class DataConfig(BaseModel):
    annotations_glob: str = "data/annotations/*.json"
    images_root: Path = Path(".")
    output_train_jsonl: Path = Path("data/finetune/train.jsonl")
    output_val_jsonl: Path = Path("data/finetune/val.jsonl")
    split_strategy: Literal["by_document"] = "by_document"
    val_ratio: float = 0.1
    include_empty_pages: bool = False
    bbox_policy: Literal["include_if_present", "drop_all"] = "include_if_present"
    bbox_space: Literal["pixel", "normalized_1000"] = "pixel"
    target_schema: Literal["finetree_exact_json"] = "finetree_exact_json"
    sample_granularity: Literal["page"] = "page"

    @model_validator(mode="after")
    def _validate_ratio(self) -> "DataConfig":
        if not (0.0 <= self.val_ratio < 1.0):
            raise ValueError("data.val_ratio must be in [0, 1).")
        return self


class PromptConfig(BaseModel):
    use_custom_prompt: bool = True
    prompt_path: Path = Path("prompt.txt")
    fallback_template: str = "Extract page JSON using the FineTree schema."


class AdapterConfig(BaseModel):
    mode: Literal["lora", "qlora"] = "lora"
    r: int = 32
    alpha: int = 64
    dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: str = "auto"
    use_rslora: bool = False
    gradient_checkpointing: str = "unsloth"


class TrainingConfig(BaseModel):
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    num_train_epochs: float = 2.0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    bf16: bool = True
    fp16: bool = False
    report_to: str = "none"


class InfraConfig(BaseModel):
    single_gpu_offload: bool = True
    enable_model_cpu_offload: bool = True
    dataloader_num_workers: int = 2


class PushToHubConfig(BaseModel):
    enabled: bool = False
    repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    hf_token_env: str = "FINETREE_HF_TOKEN"
    default_artifact: Literal["adapters_only", "merged_only", "both"] = "adapters_only"

    @model_validator(mode="after")
    def _validate_enabled(self) -> "PushToHubConfig":
        if self.enabled and not self.repo_id:
            raise ValueError("push_to_hub.repo_id is required when push_to_hub.enabled is true.")
        if self.enabled and not resolve_hf_token(self):
            raise ValueError(
                "No Hugging Face token found. Set push_to_hub.hf_token or export one of: "
                "FINETREE_HF_TOKEN, HF_TOKEN, HUGGINGFACE_HUB_TOKEN, HUGGINGFACEHUB_API_TOKEN."
            )
        return self


class InferenceConfig(BaseModel):
    model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    parser_mode: Literal["streaming_page_extraction"] = "streaming_page_extraction"
    max_new_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 0.95
    do_sample: bool = False
    torch_dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    device_map: str = "auto"
    trust_remote_code: bool = True
    load_in_4bit: bool = False


class FinetuneConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)
    push_to_hub: PushToHubConfig = Field(default_factory=PushToHubConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


def resolve_hf_token(push_cfg: PushToHubConfig) -> Optional[str]:
    candidates = [
        push_cfg.hf_token,
        os.getenv(push_cfg.hf_token_env),
        os.getenv("HF_TOKEN"),
        os.getenv("HUGGINGFACE_HUB_TOKEN"),
        os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    ]
    for token in candidates:
        if isinstance(token, str) and token.strip():
            return token.strip()
    return None


def _read_yaml_file(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to read fine-tune config files. Install with `pip install pyyaml`."
        ) from exc

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML object at top level: {path}")
    return raw


def load_finetune_config(config_path: Path | str) -> FinetuneConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fine-tune config not found: {path}")
    payload = _read_yaml_file(path)
    try:
        cfg = FinetuneConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid fine-tune config at {path}:\n{exc}") from exc

    # Resolve common relative paths relative to cwd (matches existing app conventions).
    cfg.data.images_root = cfg.data.images_root.expanduser()
    cfg.data.output_train_jsonl = cfg.data.output_train_jsonl.expanduser()
    cfg.data.output_val_jsonl = cfg.data.output_val_jsonl.expanduser()
    cfg.prompt.prompt_path = cfg.prompt.prompt_path.expanduser()
    cfg.run.output_dir = cfg.run.output_dir.expanduser()
    return cfg


__all__ = ["FinetuneConfig", "load_finetune_config", "resolve_hf_token"]
