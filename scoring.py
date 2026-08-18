"""Chills score and percentile, per scoring_spec.md (Felix, 2026-08-14).

This is deliberately independent of the ONNX video-recommendation model.
final_global_mlp.onnx keeps ranking the 40 stimuli for a person; this module
answers a different question, "how does this person's chills score rank
against the reference sample," using the 5-feature logistic model and the
2,647-participant reference distribution Felix supplied.
"""
import bisect
import json
import math
import os

_HERE = os.path.dirname(__file__)
_REF_PATH = os.path.join(_HERE, "chills_score_reference.json")

with open(_REF_PATH, "r", encoding="utf-8") as f:
    REFERENCE = json.load(f)

N_REFERENCE = REFERENCE["meta"]["n"]
WEIGHTS = REFERENCE["model"]["weights"]
INTERCEPT = REFERENCE["model"]["intercept"]
SORTED_SCORES = REFERENCE["distribution"]["sorted_scores"]
HISTOGRAM = REFERENCE["distribution"]["histogram"]


def chills_score(modtas: float, kamf: float, arousal: float, age_bucket: float, sexf: float) -> float:
    """score = sigmoid(b + w.MODTAS*MODTAS + w.KAMF*KAMF + w.Arousal*Arousal + w.Age*Age + w.SexF*SexF)"""
    z = (
        INTERCEPT
        + WEIGHTS["MODTAS"] * modtas
        + WEIGHTS["KAMF"] * kamf
        + WEIGHTS["Arousal"] * arousal
        + WEIGHTS["Age"] * age_bucket
        + WEIGHTS["SexF"] * sexf
    )
    return 1.0 / (1.0 + math.exp(-z))


def percentile_rank(score: float) -> float:
    """percentile = 100 * count(reference_scores < score) / n. 'Top X%' = 100 - this."""
    below = bisect.bisect_left(SORTED_SCORES, score)
    return 100.0 * below / N_REFERENCE


def top_percent(score: float) -> float:
    return 100.0 - percentile_rank(score)


def age_bucket(age_years: float) -> int:
    if age_years < 25:
        return 1
    if age_years < 35:
        return 2
    if age_years < 45:
        return 3
    if age_years < 55:
        return 4
    if age_years < 65:
        return 5
    return 6


def sexf_from_gender(gender: str) -> float:
    g = (gender or "").strip().lower()
    if g == "female":
        return 1.0
    if g == "male":
        return 0.0
    return 0.5


def histogram_heights_and_bin(score: float):
    """Normalized bar heights (0-1) for the reference histogram, plus which bin `score` falls in."""
    counts = HISTOGRAM["counts"]
    edges = HISTOGRAM["bin_edges"]
    peak = max(counts) if counts else 1
    heights = [c / peak for c in counts]
    bin_idx = min(len(counts) - 1, max(0, bisect.bisect_right(edges, score) - 1))
    return heights, bin_idx
