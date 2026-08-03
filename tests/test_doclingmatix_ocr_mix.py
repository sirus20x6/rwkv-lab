import importlib.util
import math
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_doclingmatix_ocr_mix.py"
SPEC = importlib.util.spec_from_file_location("doclingmatix_ocr_mix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_target_ocr_rows_is_final_mixture_share():
    rows = MODULE.target_ocr_rows(40_000, 0.10)
    assert rows == 4_445
    assert rows / (40_000 + rows) >= 0.10
    assert (rows - 1) / (40_000 + rows - 1) < 0.10


def test_doctag_to_text_removes_layout_syntax_and_duplicate_lines():
    value = (
        "<doctag><section_header_level_1><loc_1><loc_2>Invoice</section_header_level_1>"
        "<text><loc_3><loc_4>Total: $12.00</text>"
        "<text><loc_3><loc_4>Total: $12.00</text></doctag>")
    assert MODULE.doctag_to_text(value) == "Invoice\nTotal: $12.00"


def test_conversion_target_requires_clean_complete_docling_turn():
    turns = [
        {"user": "What is shown?", "assistant": "An invoice."},
        {"user": "Convert this page to docling.",
         "assistant": "<doctag><text><loc_1>" + "visible words " * 20
                      + "</text></doctag>"},
    ]
    text = MODULE.conversion_target(turns, min_chars=100, max_chars=1000)
    assert text is not None
    assert "<" not in text and "loc_" not in text
    assert MODULE.conversion_target(turns, min_chars=1000, max_chars=2000) is None


def test_located_blocks_retain_four_coordinate_contract():
    value = (
        "<doctag>"
        "<text><loc_25><loc_100><loc_475><loc_125>Invoice number 123</text>"
        "<picture><loc_0><loc_0><loc_500><loc_500></picture>"
        "<text><loc_90><loc_80><loc_20><loc_100>inverted</text>"
        "</doctag>"
    )
    assert MODULE.located_text_blocks(
        value, min_chars=4, max_chars=100) == [{
            "kind": "text", "box_500": (25, 100, 475, 125),
            "text": "Invoice number 123",
        }]


def test_glyph_heavy_conversion_target_is_rejected():
    turns = [{
        "user": "Convert this page to docling",
        "assistant": "<doctag><text>" + "GLYPH(cmap:aa) " * 3
                     + "ordinary visible text " * 20 + "</text></doctag>",
    }]
    assert MODULE.conversion_target(turns, min_chars=10, max_chars=2000) is None


def test_crop_scale_enlarges_small_text_and_never_shrinks_for_the_target():
    # A short line is enlarged to exactly the target height.
    assert MODULE.crop_scale(300, 40, target_height=128, max_edge=1536) == 3.2
    # A paragraph already taller than the target keeps its own pixels; the old
    # min() sent this to 128/400 = 0.32 and destroyed the glyphs.
    assert MODULE.crop_scale(900, 400, target_height=128, max_edge=1536) == 1.0
    # Never below 1.0 as long as the max-edge budget is not the binding limit.
    for width, height in ((900, 400), (1200, 130), (700, 129), (64, 128)):
        assert MODULE.crop_scale(
            width, height, target_height=128, max_edge=1536) >= 1.0


def test_crop_scale_respects_the_max_edge_budget_in_both_directions():
    # Oversized crops must still shrink: max_edge is a hard encoder budget.
    assert MODULE.crop_scale(
        3072, 400, target_height=128, max_edge=1536) == 0.5
    # A very wide line cannot be enlarged toward the target without breaking
    # the cap, so the cap wins and the result stays under 1.0.
    scale = MODULE.crop_scale(1600, 40, target_height=128, max_edge=1536)
    assert scale == 1536 / 1600
    assert round(1600 * scale) == 1536
    # Enlargement is capped short of the target rather than overshooting.
    scale = MODULE.crop_scale(1000, 40, target_height=128, max_edge=1536)
    assert scale == 1536 / 1000
    assert round(40 * scale) < 128


def test_additional_parents_covers_the_deficit_at_the_measured_yield():
    # 3000 rows from 2000 parents is 1.5 rows/parent, so 900 more rows need
    # 600 more parents -- not the 300 an optimistic 1+crops_per_page implies.
    assert MODULE.additional_parents(900, 3000, 2000, minimum=1) == 600
    # A fractional requirement rounds up, never down.
    assert MODULE.additional_parents(901, 3000, 2000, minimum=1) == 601
    # No deficit means no extra work, even below the minimum step.
    assert MODULE.additional_parents(0, 3000, 2000) == 0
    assert MODULE.additional_parents(-5, 3000, 2000) == 0
    # Tiny deficits still advance by a whole step so the loop cannot crawl.
    assert MODULE.additional_parents(1, 3000, 2000, minimum=256) == 256
    # Degenerate zero-yield falls back to one row per parent (pessimistic).
    assert MODULE.additional_parents(500, 0, 100, minimum=1) == 500
    assert MODULE.additional_parents(500, 100, 0, minimum=1) == 500


def test_first_tranche_reserve_matches_the_optimistic_row_estimate():
    # Reproduces the shipped invocation: 10618 train + 96 eval rows at a
    # maximum of 1 page + 2 crops per parent.
    wanted, expected_rows_per_parent, reserve_rows = 10_618 + 96, 3, 2048
    minimum_parents = math.ceil(wanted / expected_rows_per_parent)
    assert minimum_parents == 3572
    assert minimum_parents + reserve_rows == 5620
    # 5620 parents only reach 10714 rows at 1.907 rows/parent; below that the
    # tranche must be extended rather than aborting after materialization.
    observed_rows = round(5620 * 1.6)
    deficit = wanted - observed_rows
    assert deficit > 0
    extra = MODULE.additional_parents(deficit, observed_rows, 5620, minimum=1)
    assert extra == 1077
    assert round((5620 + extra) * 1.6) >= wanted


def test_split_ocr_rows_keeps_parent_pages_on_one_side_of_the_boundary():
    candidates = [{"key": f"p{index}"} for index in range(4)]
    rows = []
    for index in range(4):
        rows.append({"docling_parent_key": f"p{index}", "ocr_granularity": "page",
                     "image_sha256": f"page{index}"})
        rows.append({"docling_parent_key": f"p{index}",
                     "ocr_granularity": "located_crop",
                     "image_sha256": f"crop{index}"})
    train, held_out = MODULE.split_ocr_rows(
        rows, candidates, eval_count=2, train_count=4)
    assert len(held_out) == 2 and len(train) == 4
    train_parents = {row["docling_parent_key"] for row in train}
    eval_parents = {row["docling_parent_key"] for row in held_out}
    assert eval_parents == {"p0"}
    assert train_parents == {"p1", "p2"}
    assert not train_parents & eval_parents
    digests = [row["image_sha256"] for row in (*train, *held_out)]
    assert len(digests) == len(set(digests))


def test_image_payload_requires_exactly_one_embedded_image():
    assert MODULE.image_payload([{"bytes": b"image", "path": None}]) == b"image"
    for value in ([], [{"bytes": b"a"}, {"bytes": b"b"}], None):
        try:
            MODULE.image_payload(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted ambiguous image payload: {value!r}")
