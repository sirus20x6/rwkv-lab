from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_captioning_civitai_joy_mix",
    ROOT / "scripts/build_captioning_civitai_joy_mix.py",
)
assert SPEC is not None and SPEC.loader is not None
MIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIX)


def test_civitai_cleaning_removes_headers_negative_and_generationisms():
    value = (
        "Positive prompt: masterpiece, master_piece, 8K HDR, a naked woman "
        "holding a spear, ultra detailed\nNegative prompt: blurry, extra arms"
    )
    cleaned = MIX.clean_civitai_caption(value)
    assert cleaned == "a naked woman holding a spear"
    assert not MIX.GENERATIONISM.search(cleaned)


def test_caption_repeat_cap_is_deterministic_and_bounded():
    rows = ([{"image": f"same-{index}.jpg", "text": "same"}
             for index in range(10)]
            + [{"image": "other.jpg", "text": "other"}])
    first_stats, second_stats = Counter(), Counter()
    first = MIX.cap_caption_repeats(rows, 4, 7, first_stats)
    second = MIX.cap_caption_repeats(rows, 4, 7, second_stats)
    assert first == second
    assert Counter(row["text"] for row in first) == {"same": 4, "other": 1}
    assert first_stats["repeat_cap_removed"] == 6
    assert first_stats["output_after_repeat_cap"] == 5


def test_civitai_meta_instruction_filter_catches_prompt_artifacts():
    for value in (
        "make this much shorter for new civitai, 1girl",
        "STYLE REPLICATION: anime illustration, a person outdoors",
        "Generate an image of a person outdoors",
        "embedding:urn:air:sd1:embedding:civitai:123, a person",
    ):
        assert MIX.CIVITAI_META_INSTRUCTION.search(value)
