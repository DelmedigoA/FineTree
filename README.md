# Qwen3.5-27B LoRA Training on a Fresh RunPod Pod

## Approach

Use the base RunPod image exactly as provided for Python, CUDA, and PyTorch. Do not create a virtual environment, do not use `uv`, and do not reinstall `torch`, `torchvision`, or `torchaudio`.

Previous failures came from two issues:

- mixed Python environments
- incorrect pod-level GPU visibility, especially `NVIDIA_VISIBLE_DEVICES=void`

This procedure keeps installs minimal, verifies CUDA before installing anything, and keeps outputs under `/workspace` so they persist.

## Ordered Steps

### 1. Fix pod-level GPU environment variables in RunPod

In the RunPod pod settings:

- remove `NVIDIA_VISIBLE_DEVICES` if it is set to `void`
- remove `CUDA_VISIBLE_DEVICES` if it is blank or incorrectly set
- save the settings
- restart the pod

If `NVIDIA_VISIBLE_DEVICES=void` is present, CUDA initialization can fail even when GPUs are physically attached. Deleting old environments does not fix that.

### 2. Verify GPU access before installing anything

Run this exact block in the terminal:

```bash
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
env | egrep 'CUDA|NVIDIA' | sort

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
print("bf16:", torch.cuda.is_bf16_supported())
PY
```

Stop here if this fails. Do not continue to package installation until CUDA is working and both GPUs are visible.

### 3. Install only the higher-level packages

Do not install any `torch` packages.

```bash
export PIP_ROOT_USER_ACTION=ignore

python3 -m pip install --upgrade pip

python3 -m pip install \
  ipykernel \
  ipywidgets \
  jupyterlab_widgets \
  transformers==5.2.0 \
  datasets==3.6.0 \
  accelerate==1.13.0 \
  trl==0.28.0 \
  peft==0.18.1 \
  bitsandbytes==0.49.2 \
  huggingface_hub==1.7.0 \
  qwen-vl-utils==0.0.14 \
  ms-swift==4.0.0
```

### 4. Register a Jupyter kernel

```bash
python3 -m ipykernel install --user \
  --name runpod-image \
  --display-name "Python (runpod-image)"
```

### 5. Switch to the new kernel in JupyterLab

In JupyterLab:

`Kernel → Change Kernel → Python (runpod-image)`

Then restart the kernel.

### 6. Run notebook verification cells

First verification cell:

```python
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("device 0:", torch.cuda.get_device_name(0))
print("device 1:", torch.cuda.get_device_name(1))
print("bf16:", torch.cuda.is_bf16_supported())
```

Second verification cell:

```python
import importlib.metadata as md
import transformers, datasets, accelerate, trl, peft, bitsandbytes
import huggingface_hub, swift, ipywidgets, qwen_vl_utils

print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
print("trl:", trl.__version__)
print("peft:", peft.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("ipywidgets:", ipywidgets.__version__)
print("swift module:", swift.__file__)
print("swift version:", getattr(swift, "__version__", "unknown"))
print("qwen-vl-utils:", md.version("qwen-vl-utils"))
```

If these checks do not pass, stop and fix the environment before training.

## Training Command

Use this exact starting command:

```bash
MAX_PIXELS=1000000 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
  --model Qwen/Qwen3.5-27B \
  --tuner_type lora \
  --use_hf true \
  --dataset asafd60/FineTree_V2-approved-no-bbox-train \
  --val_dataset asafd60/FineTree_V2-approved-no-bbox-validation \
  --load_from_cache_file true \
  --freeze_vit False \
  --vit_lr 5e-6 \
  --freeze_aligner False \
  --enable_thinking false \
  --add_non_thinking_prefix true \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 16 \
  --target_modules all-linear \
  --gradient_accumulation_steps 4 \
  --output_dir /workspace/output/Qwen3.5-27B \
  --eval_steps 5 \
  --save_steps 5 \
  --save_total_limit 3 \
  --logging_steps 5 \
  --max_length 7000 \
  --warmup_ratio 0.03 \
  --dataset_num_proc 3 \
  --dataloader_num_workers 3 \
  --eval_on_start true \
  --max_pixels 1000000 \
  --temperature 0 \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --truncation_strategy right
```

## Batch Guidance

Start with:

- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=4`

With 2 GPUs, the effective batch size is 16.

If memory is comfortable, increase only:

- `gradient_accumulation_steps=8`

That gives an effective batch size of 32.

## Why Similar GPU Memory Usage Is Expected

In standard DDP or data-parallel training, each GPU holds a full copy of the model weights. Because of that, similar memory usage on both GPUs is expected. It is normal behavior and not a bug.

## Troubleshooting

- Do not create a venv.
- Do not use `uv`.
- Do not reinstall `torch`.
- Do not mix system `torch` with venv-installed `torch` or `torchvision`.
- If CUDA verification fails, check pod-level environment variables first.
- If `NVIDIA_VISIBLE_DEVICES=void` appears, remove it in RunPod settings and restart the pod.

## Benchmark

Use the benchmark UI for persisted evaluation reports and the leaderboard:

```bash
finetree-benchmark --config benchmark/config.example.yaml --host 127.0.0.1 --port 8123
```

For file-backed benchmark submissions on the benchmark machine:

```bash
finetree-benchmark-run --config benchmark/config.example.yaml --submission /path/to/run/benchmark_submission
```

The Colab notebook produces the submission bundle separately. See [benchmark/README.md](benchmark/README.md) for the unpacked folder layout and the `info.json` flow.
