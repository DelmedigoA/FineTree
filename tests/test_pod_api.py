from __future__ import annotations

from pathlib import Path

import pytest

from finetree_annotator.deploy import pod_api


def test_extract_prompt_and_image_url_supports_openai_multimodal_shape() -> None:
    prompt, image_url = pod_api._extract_prompt_and_image_url(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                }
            ]
        }
    )
    assert prompt == "Describe this"
    assert image_url == "data:image/png;base64,AAAA"


def test_extract_prompt_and_image_url_rejects_missing_image() -> None:
    with pytest.raises(ValueError):
        pod_api._extract_prompt_and_image_url(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ]
            }
        )


def test_image_path_from_data_uri_writes_file(tmp_path: Path) -> None:
    path = pod_api._image_path_from_data_uri("data:image/png;base64,AAAA", temp_dir=tmp_path)
    assert path.is_file()
    assert path.read_bytes() == b"\x00\x00\x00"


def test_bearer_token_parsing() -> None:
    assert pod_api._bearer_token("Bearer abc") == "abc"
    assert pod_api._bearer_token("bearer xyz") == "xyz"
    assert pod_api._bearer_token("Token xyz") == ""
