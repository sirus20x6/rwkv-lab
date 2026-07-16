from __future__ import annotations

import importlib.util
from pathlib import Path


def load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_vision_finish_mix.py"
    spec = importlib.util.spec_from_file_location("test_vision_finish_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_metadata_removes_domains_and_blank_sections():
    builder = load_builder()
    text = builder.cleaned_tag_text({"text": (
        "Tags: sclip; bangbros.com; brunette; outdoor\n"
        "Categories: Amateur; Outdoor\nCast:\n"
    )})
    assert text == "Tags: brunette; outdoor\nCategories: Amateur; Outdoor"


def test_joy_social_copy_is_not_accepted_as_grounded_caption():
    builder = load_builder()
    assert builder.JOY_CAPTION_SPAM.search(
        "Check out this amazing picture! Follow the artist for more. #DigitalArt 🎨")
    assert not builder.JOY_CAPTION_SPAM.search(
        "A brown dog lies on a blue sofa beside a sunlit window.")
