from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

from finetree_annotator.finetune import push_dataset_hub as push_mod


def _install_fake_qwen_vl_utils(monkeypatch, scale: float = 0.5) -> None:
    vision_mod = types.ModuleType("qwen_vl_utils.vision_process")

    def _smart_resize(height: int, width: int, **kwargs):
        _ = kwargs
        return max(1, int(round(height * scale))), max(1, int(round(width * scale)))

    vision_mod.smart_resize = _smart_resize
    root_mod = types.ModuleType("qwen_vl_utils")
    root_mod.vision_process = vision_mod
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", root_mod)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils.vision_process", vision_mod)


def _install_fake_qwen_vl_utils_with_required_factor(monkeypatch) -> None:
    vision_mod = types.ModuleType("qwen_vl_utils.vision_process")

    def _smart_resize(height: int, width: int, factor: int, min_pixels=None, max_pixels=None):
        _ = height, width, min_pixels, max_pixels
        assert factor == 28
        return 120, 240

    vision_mod.smart_resize = _smart_resize
    root_mod = types.ModuleType("qwen_vl_utils")
    root_mod.vision_process = vision_mod
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", root_mod)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils.vision_process", vision_mod)


def _empty_dataset() -> DatasetDict:
    features = Features(
        {
            "image": Image(),
            "instruction": Value("string"),
            "text": Value("string"),
        }
    )
    empty = {"image": [], "instruction": [], "text": []}
    return DatasetDict(
        {
            "train": Dataset.from_dict(empty, features=features),
            "validation": Dataset.from_dict(empty, features=features),
        }
    )


def _train_only_dataset() -> DatasetDict:
    features = Features(
        {
            "image": Image(),
            "instruction": Value("string"),
            "text": Value("string"),
        }
    )
    train = Dataset.from_dict(
        {
            "image": [str(Path("/tmp/train_img.png"))],
            "instruction": ["i"],
            "text": ["{}"],
        },
        features=features,
    )
    val = Dataset.from_dict({"image": [], "instruction": [], "text": []}, features=features)
    return DatasetDict({"train": train, "validation": val})


def test_export_for_hf_rewrites_image_paths_resizes_and_scales_bbox(tmp_path: Path, monkeypatch) -> None:
    _install_fake_qwen_vl_utils(monkeypatch, scale=0.5)

    root = tmp_path
    img_dir = root / "data" / "pdf_images" / "doc1"
    img_dir.mkdir(parents=True)
    image_path = img_dir / "page_0001.png"
    PILImage.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)

    finetune_dir = root / "data" / "finetune"
    finetune_dir.mkdir(parents=True)
    payload = {
        "meta": {"type": "other"},
        "facts": [
            {
                "bbox": {"x": 10, "y": 20, "w": 30, "h": 40},
                "value": "10",
                "refference": "",
                "path": ["a"],
            }
        ],
    }
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": f"Prompt for {image_path}"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(payload)}]},
        ],
        "metadata": {"document_id": "doc1"},
    }
    (finetune_dir / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (finetune_dir / "val.jsonl").write_text("", encoding="utf-8")

    export_dir = root / "artifacts" / "hf_dataset_export"
    train_rows, val_rows = push_mod.export_for_hf(root, export_dir, max_pixels=100_000)

    assert train_rows == 1
    assert val_rows == 0
    out = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").strip())
    assert out["image"] == "images/doc1/page_0001.png"
    assert "Prompt for" in out["instruction"]

    with PILImage.open(export_dir / out["image"]) as img:
        assert img.size == (100, 50)

    out_payload = json.loads(out["text"])
    bbox = out_payload["facts"][0]["bbox"]
    assert bbox == [5, 10, 15, 20]


def test_export_for_hf_bbox_quantization_uses_floor_for_xy_and_ceil_for_wh(tmp_path: Path, monkeypatch) -> None:
    _install_fake_qwen_vl_utils(monkeypatch, scale=0.33)

    root = tmp_path
    img_dir = root / "data" / "pdf_images" / "doc1"
    img_dir.mkdir(parents=True)
    image_path = img_dir / "page_0001.png"
    PILImage.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)

    finetune_dir = root / "data" / "finetune"
    finetune_dir.mkdir(parents=True)
    payload = {"meta": {"type": "other"}, "facts": [{"bbox": {"x": 10, "y": 11, "w": 30, "h": 31}}]}
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": "Prompt"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(payload)}]},
        ],
    }
    (finetune_dir / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (finetune_dir / "val.jsonl").write_text("", encoding="utf-8")

    export_dir = root / "artifacts" / "hf_dataset_export"
    train_rows, _ = push_mod.export_for_hf(root, export_dir, max_pixels=100_000)
    assert train_rows == 1

    out = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").strip())
    out_payload = json.loads(out["text"])
    bbox = out_payload["facts"][0]["bbox"]
    assert bbox == [3, 3, 10, 11]
    assert all(isinstance(v, int) for v in bbox)


def test_export_for_hf_minimal_instruction_is_fixed_line(tmp_path: Path) -> None:
    root = tmp_path
    img_dir = root / "data" / "pdf_images" / "doc1"
    img_dir.mkdir(parents=True)
    image_path = img_dir / "page_0001.png"
    PILImage.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)

    finetune_dir = root / "data" / "finetune"
    finetune_dir.mkdir(parents=True)
    payload = {"meta": {"type": "other"}, "facts": [{"bbox": {"x": 10, "y": 20, "w": 30, "h": 40}}]}
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": f"Long instruction with {image_path} path and bbox mentions."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(payload)}]},
        ],
    }
    (finetune_dir / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (finetune_dir / "val.jsonl").write_text("", encoding="utf-8")

    export_dir = root / "artifacts" / "hf_dataset_export"
    train_rows, _ = push_mod.export_for_hf(root, export_dir, instruction_mode="minimal")
    assert train_rows == 1
    out = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").strip())
    assert out["instruction"] == push_mod._MINIMAL_INSTRUCTION


def test_export_for_hf_compact_tokens_preserves_schema_and_compacts_payload(tmp_path: Path) -> None:
    root = tmp_path
    img_dir = root / "data" / "pdf_images" / "doc1"
    img_dir.mkdir(parents=True)
    image_path = img_dir / "page_0001.png"
    PILImage.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)

    finetune_dir = root / "data" / "finetune"
    finetune_dir.mkdir(parents=True)
    payload = {
        "meta": {"entity_name": "X", "page_num": "8", "title": "T", "type": "cash_flow"},
        "facts": [
            {
                "bbox": {"x": 10, "y": 20, "w": 30, "h": 40},
                "value": "1,234",
                "date": "30.09.2021",
                "currency": "ILS",
                "scale": 1000,
                "value_type": "amount",
                "refference": None,
                "note": None,
            }
        ],
    }
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": "Prompt"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(payload)}]},
        ],
    }
    (finetune_dir / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (finetune_dir / "val.jsonl").write_text("", encoding="utf-8")

    export_dir = root / "artifacts" / "hf_dataset_export"
    train_rows, _ = push_mod.export_for_hf(root, export_dir, compact_tokens=True)
    assert train_rows == 1

    out = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").strip())
    compact_text = out["text"]
    assert ": " not in compact_text
    assert ", " not in compact_text

    compact_payload = json.loads(compact_text)
    assert "m" in compact_payload
    assert "f" in compact_payload
    assert "meta" not in compact_payload
    assert "facts" not in compact_payload
    assert compact_payload["m"]["e"] == "X"
    assert compact_payload["m"]["p"] == "8"
    assert compact_payload["m"]["ttl"] == "T"
    fact = compact_payload["f"][0]
    assert fact["b"] == [10, 20, 30, 40]
    assert fact["v"] == "1234"
    assert fact["d"] == "30.09.2021"
    assert fact["cur"] == "ILS"
    assert fact["sc"] == 1000
    assert fact["vt"] == "amount"
    assert "ref" in fact
    assert fact["ref"] is None
    assert "note" in fact
    assert fact["note"] is None


def test_smart_resize_dimensions_supports_required_factor_signature(monkeypatch) -> None:
    _install_fake_qwen_vl_utils_with_required_factor(monkeypatch)
    h, w = push_mod._smart_resize_dimensions(500, 1000, min_pixels=200_000, max_pixels=300_000)
    assert (h, w) == (120, 240)


def test_resolve_resize_bounds_validation() -> None:
    assert push_mod._resolve_resize_bounds(None, None) is None
    assert push_mod._resolve_resize_bounds(100, 200) == (100, 200)


def test_resolve_resize_bounds_rejects_invalid_values() -> None:
    try:
        push_mod._resolve_resize_bounds(200, 100)
    except ValueError as exc:
        assert "cannot be greater" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for min > max")


def test_export_for_hf_excludes_doc_ids(tmp_path: Path, monkeypatch) -> None:
    _install_fake_qwen_vl_utils(monkeypatch, scale=0.5)
    root = tmp_path
    finetune_dir = root / "data" / "finetune"
    finetune_dir.mkdir(parents=True)

    rows = []
    for doc_id in ("pdf_3", "pdf_4"):
        img_dir = root / "data" / "pdf_images" / doc_id
        img_dir.mkdir(parents=True, exist_ok=True)
        image_path = img_dir / "page_0001.png"
        PILImage.new("RGB", (200, 100), color=(255, 255, 255)).save(image_path)
        payload = {
            "meta": {"type": "other"},
            "facts": [{"bbox": {"x": 10, "y": 10, "w": 30, "h": 20}, "value": "1", "refference": "", "path": []}],
        }
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": "Prompt"},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": json.dumps(payload)}]},
                ]
            }
        )

    (finetune_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (finetune_dir / "val.jsonl").write_text("", encoding="utf-8")

    export_dir = root / "artifacts" / "hf_dataset_export"
    train_rows, val_rows = push_mod.export_for_hf(
        root,
        export_dir,
        max_pixels=100_000,
        exclude_doc_ids={"pdf_4"},
    )
    assert train_rows == 1
    assert val_rows == 0
    out = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").strip())
    assert out["image"].startswith("images/pdf_3/")


def test_push_to_hf_drops_empty_splits(monkeypatch) -> None:
    pushed: dict[str, object] = {}

    class _DummyApi:
        def __init__(self, token: str):
            _ = token

        def create_repo(self, repo_id: str, repo_type: str, private: bool, exist_ok: bool):
            pushed["repo"] = (repo_id, repo_type, private, exist_ok)

    def _fake_push_to_hub(self: Dataset, *, repo_id: str, token: str, private: bool, split: str):
        _ = self, repo_id, token, private
        pushed.setdefault("splits", []).append(split)

    monkeypatch.setattr(push_mod, "HfApi", _DummyApi)
    monkeypatch.setattr(Dataset, "push_to_hub", _fake_push_to_hub)

    ds = _train_only_dataset()
    repo = push_mod.push_to_hf(ds, token="tok", repo_id="asafd60/FineTree-annotated-pages", private=False)
    assert repo == "asafd60/FineTree-annotated-pages"
    assert pushed["splits"] == ["train"]


def test_push_train_validation_separately_uses_train_split_for_both_repos(monkeypatch) -> None:
    pushed: dict[str, object] = {"repos": [], "splits": []}

    class _DummyApi:
        def __init__(self, token: str):
            _ = token

        def create_repo(self, repo_id: str, repo_type: str, private: bool, exist_ok: bool):
            _ = repo_type, private, exist_ok
            pushed["repos"].append(repo_id)

    def _fake_push_to_hub(self: Dataset, *, repo_id: str, token: str, private: bool, split: str):
        _ = self, token, private
        pushed["splits"].append((repo_id, split))

    monkeypatch.setattr(push_mod, "HfApi", _DummyApi)
    monkeypatch.setattr(Dataset, "push_to_hub", _fake_push_to_hub)

    ds = _empty_dataset()
    ds["train"] = Dataset.from_dict(
        {
            "image": [str(Path("/tmp/train_img.png"))],
            "instruction": ["i"],
            "text": ["{}"],
        },
        features=ds["train"].features,
    )
    ds["validation"] = Dataset.from_dict(
        {
            "image": [str(Path("/tmp/val_img.png"))],
            "instruction": ["i"],
            "text": ["{}"],
        },
        features=ds["validation"].features,
    )

    result = push_mod.push_train_validation_separately(
        ds,
        token="tok",
        base_repo_id="asafd60/FineTree-annotated-pages",
        private=False,
    )
    assert result == {
        "train": "asafd60/FineTree-annotated-pages-train",
        "validation": "asafd60/FineTree-annotated-pages-validation",
    }
    assert pushed["repos"] == [
        "asafd60/FineTree-annotated-pages-train",
        "asafd60/FineTree-annotated-pages-validation",
    ]
    assert pushed["splits"] == [
        ("asafd60/FineTree-annotated-pages-train", "train"),
        ("asafd60/FineTree-annotated-pages-validation", "train"),
    ]


def test_main_uses_export_dataset_for_push(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_build_dataset(config_path: Path) -> None:
        captured["config"] = str(config_path)

    def _fake_export(
        root: Path,
        export_dir: Path,
        *,
        instruction_mode: str,
        min_pixels=None,
        max_pixels=None,
        exclude_doc_ids=None,
        compact_tokens: bool = False,
    ):
        _ = root, instruction_mode, min_pixels, max_pixels, exclude_doc_ids
        captured["compact_tokens"] = compact_tokens
        export_dir.mkdir(parents=True, exist_ok=True)
        captured["export_dir"] = str(export_dir)
        return 1, 0

    def _fail_build_hf_dataset(root: Path):
        _ = root
        raise AssertionError("build_hf_dataset should not be called")

    def _fake_build_from_export(export_dir: Path):
        captured["build_from_export"] = str(export_dir)
        return _empty_dataset(), 1, 0

    def _fake_push(dataset: DatasetDict, token: str, repo_id: str | None, private: bool = True):
        _ = dataset, token, repo_id, private
        captured["pushed"] = True
        return "asafd60/FineTree-annotated-pages"

    monkeypatch.setattr(push_mod, "build_dataset", _fake_build_dataset)
    monkeypatch.setattr(push_mod, "export_for_hf", _fake_export)
    monkeypatch.setattr(push_mod, "build_hf_dataset", _fail_build_hf_dataset)
    monkeypatch.setattr(push_mod, "build_hf_dataset_from_export", _fake_build_from_export)
    monkeypatch.setattr(push_mod, "push_to_hf", _fake_push)

    cwd = Path.cwd()
    try:
        os_root = tmp_path
        (os_root / "configs").mkdir(parents=True)
        (os_root / "configs" / "finetune_qwen35a3_vl.yaml").write_text("{}\n", encoding="utf-8")
        monkeypatch.chdir(os_root)
        rc = push_mod.main(["--token", "tok", "--repo-id", "asafd60/FineTree-annotated-pages"])
    finally:
        monkeypatch.chdir(cwd)

    assert rc == 0
    assert captured.get("pushed") is True
    assert captured.get("build_from_export") == captured.get("export_dir")
    assert captured.get("compact_tokens") is False


def test_push_all_variants_runs_no_bbox_source_and_minimal(monkeypatch, tmp_path: Path) -> None:
    from finetree_annotator.finetune import push_dataset_hub_no_bbox as no_bbox_mod

    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(push_mod, "build_dataset", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(push_mod, "export_for_hf", lambda *_args, **_kwargs: (1, 0))
    monkeypatch.setattr(push_mod, "build_hf_dataset_from_export", lambda *_args, **_kwargs: (_empty_dataset(), 1, 0))

    def _fake_push_to_hf(dataset: DatasetDict, token: str, repo_id: str | None, private: bool = True):
        _ = dataset, token, private
        calls.append(("push_main", str(repo_id)))
        return repo_id or "auto-main"

    monkeypatch.setattr(push_mod, "push_to_hf", _fake_push_to_hf)

    def _fake_nb_export(
        root: Path,
        export_dir: Path,
        *,
        instruction_mode: str,
        min_pixels=None,
        max_pixels=None,
        exclude_doc_ids=None,
        compact_tokens: bool = False,
    ):
        _ = root, export_dir, min_pixels, max_pixels, exclude_doc_ids, compact_tokens
        calls.append(("export", instruction_mode))
        return 1, 0

    def _fake_nb_build(export_dir: Path, *, instruction_mode: str):
        _ = export_dir
        calls.append(("build", instruction_mode))
        return _empty_dataset(), 1, 0

    def _fake_nb_push(dataset: DatasetDict, token: str, repo_id: str | None, private: bool = True, *, instruction_mode: str):
        _ = dataset, token, private
        calls.append(("push", f"{instruction_mode}:{repo_id}"))
        return repo_id or "auto"

    monkeypatch.setattr(no_bbox_mod, "export_for_hf_no_bbox", _fake_nb_export)
    monkeypatch.setattr(no_bbox_mod, "build_hf_dataset_no_bbox_from_export", _fake_nb_build)
    monkeypatch.setattr(no_bbox_mod, "push_to_hf_no_bbox", _fake_nb_push)

    cwd = Path.cwd()
    try:
        os_root = tmp_path
        (os_root / "configs").mkdir(parents=True)
        (os_root / "configs" / "finetune_qwen35a3_vl.yaml").write_text("{}\n", encoding="utf-8")
        monkeypatch.chdir(os_root)
        rc = push_mod.main(
            [
                "--token",
                "tok",
                "--push-all-variants",
                "--repo-id",
                "asafd60/FineTree-annotated-pages",
                "--public",
            ]
        )
    finally:
        monkeypatch.chdir(cwd)

    assert rc == 0
    assert ("export", "source") in calls
    assert ("export", "minimal") in calls
    assert ("push_main", "asafd60/FineTree-annotated-pages-minimal-instruction") in calls
    assert ("push", "source:asafd60/FineTree-annotated-pages-no-bbox") in calls
    assert ("push", "minimal:asafd60/FineTree-annotated-pages-no-bbox-minimal-instruction") in calls
