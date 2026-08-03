import random

import pytest

from rwkv_lab.vision_ocr import (
    edit_distance,
    normalize_ocr_lines,
    normalize_ocr_text,
    ocr_generation_metrics,
    python_edit_distance,
)
from rwkv_lab.vision_train import (
    multitask_balanced_indices,
    rotate_batch,
    select_eval_sample_indices,
    task_balanced_indices,
)


def test_edit_distance_supports_characters_and_words():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance(["one", "two"], ["one", "new", "two"]) == 1
    assert edit_distance("", "abc") == 3


def test_accelerated_edit_distance_matches_the_reference_implementation():
    rng = random.Random(20260726)
    alphabet = "abcABC 0.\n"
    for _ in range(60):
        left = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        right = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert edit_distance(left, right) == python_edit_distance(left, right)
    for _ in range(20):
        left = [rng.choice(("a", "bb", "ccc")) for _ in range(rng.randint(0, 12))]
        right = [rng.choice(("a", "bb", "ccc")) for _ in range(rng.randint(0, 12))]
        assert edit_distance(left, right) == python_edit_distance(left, right)


def test_ocr_generation_metrics_are_corpus_weighted_and_report_termination():
    metrics = ocr_generation_metrics(
        [("Hello  WORLD\nLine 2", "hello world\nline 2"),
         ("abc", "axc")],
        stopped_at_eod=[True, False],
        maxed_out=[False, True],
    )
    assert metrics["examples"] == 2
    assert metrics["normalized_cer"] == pytest.approx(1 / 21)
    assert metrics["wer"] == pytest.approx(1 / 5)
    assert metrics["line_error_rate"] == pytest.approx(1 / 3)
    assert metrics["line_accuracy"] == pytest.approx(2 / 3)
    assert metrics["exact_match_rate"] == 0.5
    assert metrics["eod_rate"] == 0.5
    assert metrics["maxout_rate"] == 0.5
    # The accuracy paired with the raw CER is not reported; the emitted
    # accuracy names the normalized denominator it actually inverts.
    assert "character_accuracy" not in metrics
    assert metrics["normalized_character_accuracy"] == pytest.approx(
        1 - metrics["normalized_cer"])


def test_raw_cer_ignores_surrounding_whitespace_on_either_side():
    padded = ocr_generation_metrics([("  abc\n", "abc")])
    assert padded["cer"] == 0.0
    assert padded["reference_characters"] == 3
    assert padded["cer"] == ocr_generation_metrics([("abc", "abc")])["cer"]
    # Interior differences are still counted at full weight.
    assert ocr_generation_metrics([(" abc ", "axc")])["cer"] == pytest.approx(1 / 3)


def test_ocr_generation_metrics_validate_flag_alignment_and_empty_input():
    assert ocr_generation_metrics([]) == {}
    pairs = [("abc", "abc"), ("de", "de")]
    with pytest.raises(ValueError, match="stopped_at_eod"):
        ocr_generation_metrics(pairs, stopped_at_eod=[True])
    with pytest.raises(ValueError, match="maxed_out"):
        ocr_generation_metrics(pairs, maxed_out=[True, False, True])


def test_ocr_normalization_preserves_line_sequence_not_layout_noise():
    assert normalize_ocr_text(" A\tB\r\nC ") == "a b c"
    assert normalize_ocr_lines(" A  B \r\n\n C ") == ["a b", "c"]


def test_shuffled_image_ablation_is_a_derangement():
    assert rotate_batch(["image-a", "image-b", "image-c"]) == [
        "image-b", "image-c", "image-a"]
    with pytest.raises(ValueError):
        rotate_batch(["only-image"])


def test_task_balanced_sampler_repeats_ocr_without_dropping_rows():
    rows = [
        *[{"task": "caption", "image": f"caption-{i}"} for i in range(90)],
        *[{"task": "ocr", "image": f"ocr-{i}"} for i in range(10)],
    ]
    balanced = task_balanced_indices(
        rows, range(100), task="ocr", target_ratio=0.2, seed=7)
    assert balanced[:100] == list(range(100))
    assert len(balanced) == 113
    assert sum(rows[index]["task"] == "ocr" for index in balanced) == 23
    assert sum(rows[index]["task"] == "ocr" for index in balanced) / len(
        balanced) >= 0.2


def test_task_balancing_zero_is_resume_compatible():
    rows = [{"task": "caption"}]
    assert task_balanced_indices(
        rows, [0], task="ocr", target_ratio=0, seed=7) == [0]


def test_multitask_balancing_hits_both_targets_simultaneously():
    rows = [
        *[{"task": "caption", "image": f"caption-{i}"} for i in range(80)],
        *[{"task": "ocr", "image": f"ocr-{i}"} for i in range(10)],
        *[{"task": "sam_mask", "image": f"mask-{i}"} for i in range(10)],
    ]
    balanced = multitask_balanced_indices(
        rows, range(100),
        target_ratios={"ocr": 0.2, "sam_mask": 0.25}, seed=11)
    counts = {
        task: sum(rows[index]["task"] == task for index in balanced)
        for task in ("ocr", "sam_mask")
    }
    assert counts["ocr"] / len(balanced) >= 0.2
    assert counts["sam_mask"] / len(balanced) >= 0.25
    assert balanced[:100] == list(range(100))


def test_qualitative_gallery_reserves_ocr_for_generation_metrics():
    rows = [
        *[{"task": "caption", "source": f"caption-{i}"} for i in range(20)],
        *[{"task": "ocr", "source": "ocr"} for _ in range(8)],
        *[{"task": "sam_mask", "source": "structured"} for _ in range(8)],
    ]
    selected = select_eval_sample_indices(
        rows, range(len(rows)), 12,
        required_tasks={"ocr": 4, "structured": 4})
    tasks = [rows[index]["task"] for index in selected]
    assert tasks.count("ocr") >= 4
    assert tasks.count("sam_mask") >= 4
    assert len(selected) == len(set(selected)) == 12
