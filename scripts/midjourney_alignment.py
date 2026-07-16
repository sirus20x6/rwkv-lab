#!/usr/bin/env python3
"""Recover i1 caption suffixes for four-image Midjourney groups.

Photoroom's recap parquet preserves the four images for each numeric prompt ID,
but its duplicate-row order is not the suffix order used by i1-captions.  The
source does contain three captions aligned to each physical image.  We use
those captions to solve the four-by-four assignment to i1's five-caption rows.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence


ALIGNMENT_SCHEMA = "midjourney-caption-assignment-v1"
SOURCE_CAPTION_COLUMNS = ("qwen3", "gemini", "llava")
I1_CAPTION_COLUMNS = tuple(f"caption{i}" for i in range(1, 6))
WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.IGNORECASE)
STOPWORDS = frozenset("""
a an and are as at be been being by for from has have in into is it its of on
or that the their there this to under was were which while with image scene
depict depicts depicted showing shows features featuring presents presented
detailed overall composition foreground background visible appears rendered
lighting style artwork painting photograph photo picture view visual
""".split())


@dataclass(frozen=True)
class Alignment:
    """Mapping from recap parquet row offset to canonical i1 suffix."""

    row_to_suffix: tuple[int, int, int, int]
    score: float
    runner_up_score: float
    margin: float
    row_scores: tuple[float, float, float, float]


def _terms(text: str) -> Counter[str]:
    words = [word.lower() for word in WORD.findall(text)
             if len(word) > 2 and word.lower() not in STOPWORDS]
    terms: Counter[str] = Counter(words)
    terms.update(f"{left}_{right}" for left, right in zip(words, words[1:]))
    return terms


def _tfidf_vectors(documents: Sequence[str]) -> list[dict[str, float]]:
    counts = [_terms(document) for document in documents]
    document_frequency: Counter[str] = Counter()
    for values in counts:
        document_frequency.update(values.keys())
    total = len(documents)
    vectors = []
    for values in counts:
        weighted = {
            term: (1.0 + math.log(count))
            * (math.log((total + 1.0) / (document_frequency[term] + 1.0)) + 1.0)
            for term, count in values.items()
        }
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        vectors.append({term: value / norm for term, value in weighted.items()})
    return vectors


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def align_group(source_rows: Sequence[Sequence[str]],
                i1_rows: Sequence[Sequence[str]]) -> Alignment:
    """Align four physical source rows to four i1 caption suffixes.

    Each source row may contain Qwen3, Gemini, and LLaVA descriptions; each i1
    row may contain up to five Qwen3-VL descriptions. Empty descriptions are
    ignored. A global assignment prevents two physical images from claiming
    the same canonical suffix.
    """
    if len(source_rows) != 4 or len(i1_rows) != 4:
        raise ValueError("Midjourney alignment requires exactly four source and i1 rows")
    source_documents = ["\n".join(text.strip() for text in row if text and text.strip())
                        for row in source_rows]
    i1_documents = ["\n".join(text.strip() for text in row if text and text.strip())
                    for row in i1_rows]
    if any(not document for document in source_documents):
        raise ValueError("every Midjourney source image needs at least one aligned caption")
    # A tiny number of i1 rows are absent from the published caption parquet.
    # One absent row is still unambiguous: solve the other three assignments
    # globally and give the remaining physical image to the missing suffix.
    if sum(bool(document) for document in i1_documents) < 3:
        raise ValueError("at least three Midjourney i1 suffixes need captions")
    vectors = _tfidf_vectors([*source_documents, *i1_documents])
    scores = [[_cosine(vectors[row], vectors[4 + suffix])
               for suffix in range(4)] for row in range(4)]
    ranked = sorted(
        ((sum(scores[row][permutation[row]] for row in range(4)), permutation)
         for permutation in itertools.permutations(range(4))),
        reverse=True,
    )
    best_score, best = ranked[0]
    runner_up = ranked[1][0]
    return Alignment(
        row_to_suffix=best,
        score=best_score,
        runner_up_score=runner_up,
        margin=best_score - runner_up,
        row_scores=tuple(scores[row][best[row]] for row in range(4)),
    )
