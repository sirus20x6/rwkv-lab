from scripts.scan_midjourney_v6_caption_routes import (
    classify_caption,
    route_caption_votes,
)


def test_caption_medium_classifier_is_conservative():
    assert classify_caption("A black-and-white photograph of two women.") == "photo"
    assert classify_caption("A richly textured oil painting of a harbor.") == "animation"
    assert classify_caption("A photorealistic digital painting of a knight.") == "conflict"
    assert classify_caption("A woman standing beside a window.") == "unknown"
    assert classify_caption(None) == "missing"


def test_caption_routes_require_unopposed_agreement():
    assert route_caption_votes(["photo", "photo", "unknown"]) == "photo"
    assert (
        route_caption_votes(["animation", "animation", "unknown"]) == "animation"
    )
    assert route_caption_votes(["photo", "animation", "photo"]) == "ambiguous"
    assert route_caption_votes(["photo", "photo", "conflict"]) == "ambiguous"
    assert route_caption_votes(["photo", "unknown", "missing"]) == "ambiguous"
