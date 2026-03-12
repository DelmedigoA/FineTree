from __future__ import annotations

from finetree_annotator.model_prompt_serialization import (
    MODEL_PROMPT_MODE,
    SESSION_STORAGE_MODE,
    build_single_page_payload,
    coerce_payload_for_schema_mode,
)


def test_build_single_page_payload_uses_model_prompt_shape() -> None:
    payload = build_single_page_payload(
        page_name="page_0001.png",
        page_meta={"page_type": "other"},
        facts=[{"value": "10", "equations": [{"equation": "6 + 4", "fact_equation": "f1 + f2"}]}],
        mode=MODEL_PROMPT_MODE,
    )

    assert payload == {
        "pages": [
            {
                "image": "page_0001.png",
                "meta": {"page_type": "other"},
                "facts": [{"value": "10", "equations": [{"equation": "6 + 4", "fact_equation": "f1 + f2"}]}],
            }
        ]
    }


def test_coerce_payload_for_schema_mode_drops_document_wrapper_for_model_prompts() -> None:
    payload = coerce_payload_for_schema_mode(
        {
            "images_dir": "data/pdf_images/doc1",
            "metadata": {"language": "he"},
            "pages": [{"image": "page_0001.png", "meta": {"page_type": "other"}, "facts": [{"value": "10"}]}],
        },
        mode=MODEL_PROMPT_MODE,
    )

    assert "images_dir" not in payload
    assert "metadata" not in payload
    assert payload["pages"][0]["image"] == "page_0001.png"


def test_coerce_payload_for_schema_mode_keeps_storage_wrapper() -> None:
    payload = coerce_payload_for_schema_mode(
        {"meta": {"page_type": "other"}, "facts": [{"value": "10"}]},
        mode=SESSION_STORAGE_MODE,
        default_page_name="page_0001.png",
    )

    assert payload["images_dir"] is None
    assert payload["metadata"] == {}
    assert payload["pages"][0]["image"] == "page_0001.png"
