import json
from pathlib import Path

from rwkv_lab.kimi_teacher import (
    CAPTION_PROMPT,
    build_queue,
    caption_selection_score,
    logprob_summary,
    make_payload,
    response_cost,
    write_derived_manifests,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_selection_is_length_first_with_narrow_junk_penalties(tmp_path):
    clean = "The image captures " + " ".join(f"detail{i}" for i in range(80)) + "."
    duplicated = clean + " The image shows " + " ".join(f"extra{i}" for i in range(20)) + "."
    assert caption_selection_score(clean) == 83
    # It remains longer, but does not receive full credit for a likely second
    # caption preamble.
    assert 83 < caption_selection_score(duplicated) < 103

    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    queue = tmp_path / "queue.jsonl"
    write_jsonl(train, [
        {"image": "adult.jpg", "text": "x " * 500,
         "stage1_source": "eight_hour_nsfw_video"},
        {"image": "short.jpg", "text": "a concise scene",
         "stage1_source": "eight_hour_i1_pexels"},
        {"image": "long.jpg", "text": clean,
         "stage1_source": "eight_hour_i1_pexels"},
    ])
    write_jsonl(evaluation, [
        {"image": "joy.jpg", "text": "excluded joy",
         "stage1_source": "eval_joy_caption"},
        {"image": "mj.jpg", "text": "clean evaluation scene",
         "stage1_source": "eval_i1_midjourneyv6"},
    ])
    result = build_queue(
        train, evaluation, queue,
        train_limit=2, eval_limit=10, require_images=False,
    )
    rows = [json.loads(line) for line in queue.read_text().splitlines()]
    assert result == {
        "output": str(queue), "eval": 1, "train": 2, "total": 3,
        "dropped_duplicates": 0,
        "top_train_words": 83, "bottom_train_words": 3,
    }
    assert [row["image"] for row in rows] == ["mj.jpg", "long.jpg", "short.jpg"]
    assert all("adult" not in row["image"] and "joy" not in row["image"] for row in rows)


def test_payload_preserves_full_frame_signal_and_requests_max_logprobs():
    payload = make_payload(
        "data:image/webp;base64,AAAA",
        max_completion_tokens=2048,
        top_logprobs=20,
    )
    assert payload["provider"] == {
        "only": ["decart"],
        "order": ["decart"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 20
    assert payload["max_tokens"] == 2048
    assert payload["reasoning"]["effort"] == "none"
    assert payload["messages"][0]["content"][0]["image_url"]["url"].endswith("AAAA")
    assert payload["messages"][0]["content"][1]["text"] == CAPTION_PROMPT

    unlimited = make_payload("data:image/png;base64,x", max_completion_tokens=0,
                             top_logprobs=20)
    assert "max_tokens" not in unlimited


def test_logprobs_and_usage_are_retained_as_teacher_statistics():
    response = {
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "choices": [{"logprobs": {"content": [
            {"token": "A", "logprob": -0.1,
             "top_logprobs": [{"token": "A", "logprob": -0.1}]},
            {"token": "B", "logprob": -1.0,
             "top_logprobs": [
                 {"token": "B", "logprob": -1.0},
                 {"token": "C", "logprob": -1.2},
             ]},
        ]}}],
    }
    summary = logprob_summary(response)
    assert summary["tokens"] == 2
    assert summary["sequence_logprob"] == -1.1
    assert summary["mean_logprob"] == -0.55
    assert summary["top_alternatives"] == 3
    assert response_cost(response) == 100 * 0.66e-6 + 20 * 3.41e-6


def test_derived_manifest_excludes_truncated_receipt(tmp_path):
    good = {
        "accepted": True,
        "caption": "A grounded, complete description.",
        "finish_reason": "stop",
        "cost_usd": 0.001,
        "queue": {"id": "good", "image": "good.webp", "split": "train"},
        "response": {"id": "request-good"},
        "logprob_summary": {"tokens": 5, "mean_logprob": -0.2},
    }
    truncated = {
        "accepted": False,
        "caption": "An unfinished description",
        "finish_reason": "length",
        "cost_usd": 0.002,
        "queue": {"id": "bad", "image": "bad.webp", "split": "eval"},
        "response": {"id": "request-bad"},
    }
    summary = write_derived_manifests(tmp_path, [good, truncated])
    assert summary["spent_usd"] == 0.003
    assert summary["train"] == 1
    assert summary["eval"] == 0
    row = json.loads((tmp_path / "train.jsonl").read_text())
    assert row["teacher"]["receipt"] == "raw/good.json"
    assert row["teacher"]["mean_logprob"] == -0.2


def test_multi_caption_images_collapse_to_one_queue_item(tmp_path):
    long_caption = "The scene contains " + " ".join(f"word{i}" for i in range(60))
    short_caption = "a concise scene"
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    queue = tmp_path / "queue.jsonl"
    write_jsonl(train, [
        {"image": "dup.jpg", "text": short_caption,
         "stage1_source": "eight_hour_i1_pexels"},
        {"image": "dup.jpg", "text": long_caption,
         "stage1_source": "eight_hour_i1_pexels"},
        {"image": "other.jpg", "text": "another scene entirely",
         "stage1_source": "eight_hour_i1_pexels"},
    ])
    write_jsonl(evaluation, [])
    result = build_queue(
        train, evaluation, queue,
        train_limit=0, eval_limit=0, require_images=False,
    )
    rows = [json.loads(line) for line in queue.read_text().splitlines()]
    assert result["train"] == 2
    assert result["dropped_duplicates"] == 1
    assert [row["image"] for row in rows] == ["dup.jpg", "other.jpg"]
    # The best-scoring caption is the one kept for the shared identity.
    kept = next(row for row in rows if row["image"] == "dup.jpg")
    assert kept["old_caption"] == long_caption
    assert len({row["id"] for row in rows}) == len(rows)


def test_body_level_openrouter_errors_classify_retryable(monkeypatch):
    import io

    import rwkv_lab.kimi_teacher as kimi_teacher
    from rwkv_lab.kimi_teacher import ApiFailure, post_completion

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_urlopen(body):
        payload = json.dumps(body).encode()
        return lambda request, timeout: Response(payload)

    import pytest

    for code, retryable in ((429, True), (502, True), (400, False)):
        monkeypatch.setattr(
            kimi_teacher.urllib.request, "urlopen",
            fake_urlopen({"error": {"code": code, "message": "provider says no",
                                    "metadata": {"provider_name": "x"}}}))
        with pytest.raises(ApiFailure) as info:
            post_completion({}, "key", timeout=1)
        assert info.value.retryable is retryable
    # An error body with no numeric code stays non-retryable.
    monkeypatch.setattr(kimi_teacher.urllib.request, "urlopen",
                        fake_urlopen({"error": "opaque failure"}))
    with pytest.raises(ApiFailure) as info:
        post_completion({}, "key", timeout=1)
    assert info.value.retryable is False


def test_execute_queue_requeues_local_worker_failure_once(tmp_path, monkeypatch):
    import argparse

    import rwkv_lab.kimi_teacher as kimi_teacher

    queue = tmp_path / "queue.jsonl"
    write_jsonl(queue, [{"id": "item0", "split": "train", "image": "x.jpg"}])
    calls = []

    def fake_caption_one(item, *, config, api_key):
        calls.append(item["id"])
        if len(calls) == 1:
            raise RuntimeError("transient local failure before any receipt")
        return {
            "accepted": True, "caption": "a caption", "finish_reason": "stop",
            "cost_usd": 0.0, "queue": item, "response": {"id": "r0"},
            "logprob_summary": {"tokens": 1, "mean_logprob": -0.1},
        }

    monkeypatch.setattr(kimi_teacher, "caption_one", fake_caption_one)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    args = argparse.Namespace(
        execute=True, api_key_env="OPENROUTER_API_KEY", queue=queue,
        output=tmp_path / "out", top_logprobs=20, max_items=0,
        max_completion_tokens=64, max_image_side=0, timeout=1, retries=0,
        providers=["inceptron"], no_provider_fallbacks=False, workers=1,
        budget_usd=1.0, input_reserve_usd=0.0, max_consecutive_failures=4,
    )
    summary = kimi_teacher.execute_queue(args)
    assert calls == ["item0", "item0"]
    assert summary["train"] == 1
