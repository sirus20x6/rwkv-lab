import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_civitai_anima_dataset.py"
SPEC = importlib.util.spec_from_file_location("civitai_anima_dataset", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def item(**overrides):
    value = {
        "id": 123,
        "url": "https://image.civitai.com/path/original=true/example.jpeg",
        "width": 1024,
        "height": 1024,
        "type": "image",
        "nsfwLevel": "None",
        "modelVersionIds": [2945208],
        "meta": {
            "prompt": "masterpiece, best quality, 1girl, blue hair",
            "negativePrompt": "low quality",
        },
    }
    value.update(overrides)
    return value


def test_prompt_bearing_image_is_eligible():
    eligibility, reason, prompt, negative = module.classify_image(item())
    assert eligibility == "eligible"
    assert reason == "prompt_bearing_official_gallery_image"
    assert prompt.endswith("blue hair")
    assert negative == "low quality"


def test_metadata_and_prompt_are_required():
    assert module.classify_image(item(meta=None))[:2] == (
        "excluded",
        "missing_metadata",
    )
    assert module.classify_image(item(meta={"prompt": "  "}))[:2] == (
        "excluded",
        "empty_prompt",
    )


def test_non_images_and_untrusted_urls_are_excluded():
    assert module.classify_image(item(type="video"))[:2] == (
        "excluded",
        "non_image_media",
    )
    assert module.classify_image(item(url="https://example.com/image.png"))[:2] == (
        "excluded",
        "unsafe_image_url",
    )


def test_official_version_membership_can_be_required():
    assert module.classify_image(item(), {2945208})[0] == "eligible"
    assert module.classify_image(item(modelVersionIds=[999]), {2945208})[:2] == (
        "excluded",
        "no_official_anima_version",
    )


def test_non_sfw_levels_are_quarantined():
    assert module.classify_image(item(nsfwLevel="Mature"))[:2] == (
        "quarantined",
        "content_level_Mature",
    )


def test_sexualized_minor_prompts_are_quarantined():
    value = item(
        meta={
            "prompt": "schoolgirl, nude, explicit",
            "negativePrompt": "",
        }
    )
    assert module.classify_image(value)[:2] == (
        "quarantined",
        "sexualized_minor_prompt",
    )


def test_safe_child_prompt_is_not_misclassified_as_sexual():
    value = item(
        meta={
            "prompt": "children playing in a park, safe, illustration",
            "negativePrompt": "",
        }
    )
    assert module.classify_image(value)[0] == "eligible"


def test_image_extension_is_deterministic():
    assert module.image_extension("JPEG") == ".jpg"
    assert module.image_extension("png") == ".png"
