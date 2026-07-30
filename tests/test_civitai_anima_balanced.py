import importlib.util
import json
from pathlib import Path

from PIL import Image, PngImagePlugin

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "download_civitai_anima_balanced.py"
)
SPEC = importlib.util.spec_from_file_location("civitai_anima_balanced", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


OFFICIAL = {2945208}


def item(**overrides):
    value = {
        "id": 123,
        "url": "https://image.civitai.com/path/original=true/example.jpeg",
        "width": 1024,
        "height": 1024,
        "type": "image",
        "nsfwLevel": "None",
        "modelVersionIds": [2945208],
        "stats": {
            "likeCount": 4,
            "heartCount": 3,
            "laughCount": 2,
            "cryCount": 1,
            "dislikeCount": 99,
            "commentCount": 99,
        },
        "meta": {
            "prompt": "masterpiece, best quality, 1girl, blue hair",
            "negativePrompt": "low quality",
        },
    }
    value.update(overrides)
    return value


def test_sfw_none_record_is_eligible():
    result = module.classify_image(item(), "sfw", OFFICIAL)
    assert result[:2] == ("eligible", "prompt_bearing_official_anima_sfw")


def test_adult_mature_record_is_eligible():
    result = module.classify_image(
        item(nsfwLevel="Mature", meta={"prompt": "adult woman, nude"}),
        "nsfw",
        OFFICIAL,
    )
    assert result[:2] == ("eligible", "prompt_bearing_official_anima_nsfw")


def test_adult_feed_rejects_none_level():
    result = module.classify_image(item(), "nsfw", OFFICIAL)
    assert result[:2] == ("excluded", "wrong_adult_level_None")


def test_sfw_feed_rejects_mature_level():
    result = module.classify_image(item(nsfwLevel="Mature"), "sfw", OFFICIAL)
    assert result[:2] == ("excluded", "wrong_sfw_level_Mature")


def test_any_minor_marker_is_quarantined_in_adult_feed():
    result = module.classify_image(
        item(nsfwLevel="X", meta={"prompt": "schoolgirl portrait"}),
        "nsfw",
        OFFICIAL,
    )
    assert result[:2] == ("quarantined", "minor_marker_in_adult_prompt")


def test_numeric_underage_marker_is_quarantined_in_adult_feed():
    result = module.classify_image(
        item(nsfwLevel="X", meta={"prompt": "17 years old"}),
        "nsfw",
        OFFICIAL,
    )
    assert result[:2] == ("quarantined", "minor_marker_in_adult_prompt")


def test_safe_child_scene_remains_allowed_in_sfw_feed():
    result = module.classify_image(
        item(meta={"prompt": "children playing safely in a park"}),
        "sfw",
        OFFICIAL,
    )
    assert result[0] == "eligible"


def test_sexualized_minor_is_quarantined_in_sfw_feed():
    result = module.classify_image(
        item(meta={"prompt": "schoolgirl, nude, explicit"}),
        "sfw",
        OFFICIAL,
    )
    assert result[:2] == ("quarantined", "sexualized_minor_prompt")


def test_official_version_is_required():
    result = module.classify_image(
        item(modelVersionIds=[999]), "sfw", OFFICIAL
    )
    assert result[:2] == ("excluded", "no_official_anima_version")


def test_reaction_total_uses_reactions_not_comments_or_dislikes():
    counts = module.reaction_counts(item())
    assert sum(counts.values()) == 10
    assert set(counts) == set(module.REACTION_KEYS)


def test_devalue_unflatten_decodes_ranked_response():
    flattened = [
        {"nextCursor": 1, "items": 2},
        "20|123",
        [3],
        {"id": 4, "reactionCount": 5, "hasMeta": 6},
        123456,
        42,
        True,
    ]
    decoded = module.devalue_unflatten(
        {"result": {"data": json.dumps(flattened)}}
    )
    assert decoded == {
        "nextCursor": "20|123",
        "items": [{"id": 123456, "reactionCount": 42, "hasMeta": True}],
    }


def test_uuid_is_converted_to_original_image_url():
    value = "14f3cc00-f0f6-4e65-bae6-e947b8a21bff"
    assert module.original_image_url(value).endswith(f"/original=true/{value}")


def test_corrupt_ancillary_png_crc_keeps_original_decodable_pixels(tmp_path):
    path = tmp_path / "ancillary-crc.png"
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("comment", "hello")
    Image.new("RGB", (8, 6), "red").save(path, pnginfo=png_info)
    payload = bytearray(path.read_bytes())
    marker = payload.index(b"comment\x00hello")
    payload[marker + len(b"comment\x00")] ^= 1
    path.write_bytes(payload)

    assert module.png_critical_chunks_valid(path)
    width, height, image_format, warning = module.inspect_image(path)
    assert (width, height, image_format) == (8, 6, "PNG")
    assert warning == "corrupt_ancillary_png_crc"
