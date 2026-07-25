import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_captioning_first_mix import (  # noqa: E402
    choose_grounded_caption,
    clean_grounded_caption,
    coco_sam_prompt,
    _coco_sam_target,
    _concept_sam_target,
    _mask_row_spans,
    places_key,
    select_smallest,
    stem_key,
    unique_content_rows,
)


def test_grounded_caption_drops_generator_tail_but_keeps_visible_facts():
    text = (
        "Three yellow kites fly over a dark blue ocean. "
        "The overall composition creates a sense of timeless joy and serenity."
    )
    assert clean_grounded_caption(text) == (
        "Three yellow kites fly over a dark blue ocean.")


def test_grounded_caption_selection_prefers_consensus_over_speculation():
    text, variant = choose_grounded_caption([
        ("caption1", "A red kite flies over the blue ocean. A woman may be watching."),
        ("caption2", "A red kite flies above a blue ocean under a pale sky."),
        ("caption3", "A red kite flies over the blue ocean beneath a pale sky."),
    ])
    assert variant in {"caption2", "caption3"}
    assert "woman" not in text


def test_stable_selection_is_order_independent_and_unique():
    rows = [{"i1_key": str(index)} for index in range(100)]
    expected = select_smallest(
        rows, 9, seed=17, namespace="test")
    actual = select_smallest(
        list(reversed(rows)) + rows[:5], 9, seed=17, namespace="test")
    assert actual == expected
    assert len({row["i1_key"] for row in actual}) == 9


def test_archive_key_mappings_match_i1_keys():
    assert places_key("data_large/w/windmill/00018748.jpg") == (
        "windmill_00018748")
    assert places_key("data_large/a/atrium/public/00013554.jpg") == (
        "public_00013554")
    assert stem_key("train/species/34906d98-b6d1.jpg") == (
        "34906d98-b6d1")
    assert stem_key("train/species/readme.txt") is None


def test_unique_content_backfills_duplicate_file_keys():
    rows = [
        {"i1_key": "a", "image_sha256": "same"},
        {"i1_key": "b", "image_sha256": "same"},
        {"i1_key": "c", "image_sha256": "different"},
    ]
    assert [row["i1_key"] for row in unique_content_rows(rows, 2)] == ["a", "c"]


def test_coco_polygon_becomes_compact_mask_rows_and_normalized_box():
    annotation = {
        "id": 7, "category_id": 3, "area": 2500, "bbox": [25, 25, 50, 50],
        "segmentation": [[25, 25, 75, 25, 75, 75, 25, 75]],
    }
    spans = _mask_row_spans(annotation, 100, 100)
    assert "4:4-12" in spans
    target, annotation_ids = _coco_sam_target(
        [annotation], {3: "object"}, 100, 100)
    assert "object; box=[250,250,749,749]; mask16=" in target
    assert annotation_ids == [7]


def test_coco_prompt_names_requested_categories_and_instance_counts():
    target = "\n".join([
        "chair; box=[100,100,300,300]; mask16=1:1-4",
        "chair; box=[500,100,700,300]; mask16=1:8-11",
        "person; box=[200,200,800,900]; mask16=3:3-12",
    ])
    prompt = coco_sam_prompt(target)
    assert "chair: 2 largest" in prompt
    assert "person: 1 largest" in prompt
    assert "exactly one line per requested instance" in prompt
    assert "annotated objects" not in prompt


def test_concept_mask_target_serializes_instances_without_repeating_prompt_label():
    annotation = {
        "id": 9, "area": 2500, "bbox": [25, 25, 50, 50],
        "segmentation": [[25, 25, 75, 25, 75, 75, 25, 75]],
    }
    target, annotation_ids = _concept_sam_target([annotation], 100, 100)
    assert target.startswith("instance 1; box=[250,250,749,749]; mask16=")
    assert annotation_ids == [9]
