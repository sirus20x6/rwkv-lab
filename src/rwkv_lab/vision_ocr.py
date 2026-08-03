"""OCR evaluation utilities independent of the training model."""
from __future__ import annotations

import re
from collections.abc import Sequence


WHITESPACE = re.compile(r"\s+")

try:  # Optional: a compiled Levenshtein is ~1000x faster on OCR-length strings.
    from rapidfuzz.distance import Levenshtein as _RAPIDFUZZ_LEVENSHTEIN
except ImportError:  # pragma: no cover - exercised by the pure-Python fallback
    _RAPIDFUZZ_LEVENSHTEIN = None


def python_edit_distance(left: Sequence, right: Sequence) -> int:
    """Return Levenshtein distance using O(min(len(left), len(right))) memory.

    The reference implementation. It is O(n*m) in Python and a single 3k-char
    OCR pair costs seconds, so :func:`edit_distance` prefers rapidfuzz when it
    is installed; this remains the definition both paths must agree on.
    """
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, 1):
        current = [row]
        for column, right_item in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def edit_distance(left: Sequence, right: Sequence) -> int:
    """Return unit-cost Levenshtein distance over characters or token lists."""
    if _RAPIDFUZZ_LEVENSHTEIN is not None:
        return int(_RAPIDFUZZ_LEVENSHTEIN.distance(left, right))
    return python_edit_distance(left, right)


def normalize_ocr_text(value: str) -> str:
    """Case-fold OCR text and collapse layout-only whitespace."""
    return WHITESPACE.sub(" ", value.replace("\r", "\n")).strip().casefold()


def normalize_ocr_lines(value: str) -> list[str]:
    """Normalize non-empty lines while preserving their sequence."""
    return [
        WHITESPACE.sub(" ", line).strip().casefold()
        for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    ]


def ocr_generation_metrics(
        pairs: Sequence[tuple[str, str]],
        *,
        stopped_at_eod: Sequence[bool] | None = None,
        maxed_out: Sequence[bool] | None = None,
) -> dict[str, float]:
    """Compute corpus OCR accuracy and decoding-termination diagnostics."""
    if not pairs:
        return {}
    if stopped_at_eod is not None and len(stopped_at_eod) != len(pairs):
        raise ValueError("stopped_at_eod must align with OCR pairs")
    if maxed_out is not None and len(maxed_out) != len(pairs):
        raise ValueError("maxed_out must align with OCR pairs")

    character_edits = normalized_character_edits = word_edits = line_edits = 0
    reference_characters = normalized_characters = reference_words = reference_lines = 0
    exact = 0
    for reference, prediction in pairs:
        # Both sides are stripped: callers do not agree on whether a decoded
        # sample arrives stripped, and leading/trailing layout whitespace in an
        # OCR target would otherwise score as pure insertions in the raw CER.
        reference = reference.replace("\r\n", "\n").replace("\r", "\n").strip()
        prediction = prediction.replace("\r\n", "\n").replace("\r", "\n").strip()
        character_edits += edit_distance(reference, prediction)
        reference_characters += len(reference)

        normalized_reference = normalize_ocr_text(reference)
        normalized_prediction = normalize_ocr_text(prediction)
        normalized_character_edits += edit_distance(
            normalized_reference, normalized_prediction)
        normalized_characters += len(normalized_reference)
        reference_word_list = normalized_reference.split()
        prediction_word_list = normalized_prediction.split()
        word_edits += edit_distance(reference_word_list, prediction_word_list)
        reference_words += len(reference_word_list)

        reference_line_list = normalize_ocr_lines(reference)
        prediction_line_list = normalize_ocr_lines(prediction)
        line_edits += edit_distance(reference_line_list, prediction_line_list)
        reference_lines += len(reference_line_list)
        exact += normalized_reference == normalized_prediction

    def rate(edits: int, total: int) -> float:
        return float(edits) / max(1, total)

    cer = rate(character_edits, reference_characters)
    normalized_cer = rate(normalized_character_edits, normalized_characters)
    wer = rate(word_edits, reference_words)
    line_error_rate = rate(line_edits, reference_lines)
    result = {
        "examples": float(len(pairs)),
        "reference_characters": float(reference_characters),
        "reference_words": float(reference_words),
        "reference_lines": float(reference_lines),
        "cer": cer,
        "normalized_cer": normalized_cer,
        "wer": wer,
        "line_error_rate": line_error_rate,
        # Named for its denominator: this pairs with ``normalized_cer``, not
        # with the raw ``cer`` also emitted here.
        "normalized_character_accuracy": max(0.0, 1.0 - normalized_cer),
        "word_accuracy": max(0.0, 1.0 - wer),
        "line_accuracy": max(0.0, 1.0 - line_error_rate),
        "exact_match_rate": exact / len(pairs),
    }
    if stopped_at_eod is not None:
        result["eod_rate"] = sum(bool(value) for value in stopped_at_eod) / len(pairs)
    if maxed_out is not None:
        result["maxout_rate"] = sum(bool(value) for value in maxed_out) / len(pairs)
    return result
