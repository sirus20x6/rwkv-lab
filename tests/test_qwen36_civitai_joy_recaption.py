import json
from pathlib import Path

from scripts.batch_recaption_civitai_joy_qwen36 import (
    SYSTEM_INSTRUCTION,
    caption_instruction,
    caption_messages,
    completed_job_ids,
    iter_jobs,
)


def test_instruction_preserves_language_tone_and_untrusted_hint():
    prompt = caption_instruction("une femme avec un chapeau")
    assert "same primary natural language" in SYSTEM_INSTRUCTION
    assert "Do not translate" in SYSTEM_INSTRUCTION
    assert "overly clinical" in SYSTEM_INSTRUCTION
    assert "UNTRUSTED hint" in SYSTEM_INSTRUCTION
    assert "une femme avec un chapeau" in prompt


def test_shared_prefix_and_unique_content_precede_image():
    messages = caption_messages("a red fox", "data:image/jpeg;base64,abc")
    assert messages[0] == {"role": "system", "content": SYSTEM_INSTRUCTION}
    content = messages[1]["content"]
    assert content[0]["type"] == "text"
    assert "a red fox" in content[0]["text"]
    assert content[1]["type"] == "image_url"


def test_job_inventory_filters_sources_and_is_stable(tmp_path: Path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    manifest = tmp_path / "mix_train.jsonl"
    rows = [
        {"image": str(image), "text": "hint", "stage1_source": "captioning_joy"},
        {"image": str(image), "text": "hint", "stage1_source": "captioning_joy"},
        {"image": str(image), "text": "other", "stage1_source": "captioning_i1_pexels"},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    jobs = list(iter_jobs([manifest]))
    assert len(jobs) == 1
    assert jobs[0].source == "captioning_joy"
    assert jobs[0].split == "train"


def test_resume_only_accepts_successful_nonempty_records(tmp_path: Path):
    output = tmp_path / "results.jsonl"
    rows = [
        {"job_id": "a", "status": "ok", "caption": "caption"},
        {"job_id": "b", "status": "error", "error": "retry"},
        {"job_id": "c", "status": "ok", "caption": ""},
    ]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert completed_job_ids(output) == {"a"}
