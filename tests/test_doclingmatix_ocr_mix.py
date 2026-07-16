import importlib.util
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


def test_glyph_heavy_conversion_target_is_rejected():
    turns = [{
        "user": "Convert this page to docling",
        "assistant": "<doctag><text>" + "GLYPH(cmap:aa) " * 3
                     + "ordinary visible text " * 20 + "</text></doctag>",
    }]
    assert MODULE.conversion_target(turns, min_chars=10, max_chars=2000) is None


def test_image_payload_requires_exactly_one_embedded_image():
    assert MODULE.image_payload([{"bytes": b"image", "path": None}]) == b"image"
    for value in ([], [{"bytes": b"a"}, {"bytes": b"b"}], None):
        try:
            MODULE.image_payload(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted ambiguous image payload: {value!r}")
