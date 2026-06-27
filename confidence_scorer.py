"""
Confidence Scorer — combines two detection signals into a single verdict.

Algorithm (planning.md §1 Step 5):
  1. Weighted combination:
        combined = 0.55 * llm_score + 0.45 * stylometric_score

  2. Distance-from-boundary confidence:
        confidence = 2 * |combined - 0.5|

  3. Signal disagreement cap:
        if |llm_score - stylometric_score| > 0.40 → cap confidence at 0.60

  4. Label thresholds:
        combined >= 0.70  →  "human"
        combined <= 0.30  →  "ai"
        otherwise         →  "uncertain"
"""

from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM_WEIGHT = 0.55
STYLOMETRIC_WEIGHT = 0.45
DISAGREEMENT_THRESHOLD = 0.40
DISAGREEMENT_CAP = 0.60


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _effective_weights(
    text_length: int | None,
    max_llm_weight: float = 0.70,
) -> tuple[float, float]:
    """
    Return (llm_weight, stylometric_weight) adjusted for text length.

    When text is under 100 words the stylometric metrics are unreliable
    (planning.md §3b Edge Case 4).  The LLM weight increases linearly
    from its default 0.55 to *max_llm_weight* at 50 words.

    Different content types can set different max weights via
    content_types.py (e.g., metadata gets 0.80 since it's always short).
    """
    if text_length is None or text_length >= 100:
        return LLM_WEIGHT, STYLOMETRIC_WEIGHT

    if text_length <= 50:
        return max_llm_weight, round(1.0 - max_llm_weight, 4)

    # Linear ramp from 0.55 at 100 words to max_llm_weight at 50 words
    frac = (100 - text_length) / 50.0        # 0.0 at 100w, 1.0 at 50w
    llm_w = LLM_WEIGHT + frac * (max_llm_weight - LLM_WEIGHT)
    return round(llm_w, 4), round(1.0 - llm_w, 4)


def combine(
    llm_score: float | None,
    stylometric_score: float,
    llm_weight: float = LLM_WEIGHT,
    stylometric_weight: float = STYLOMETRIC_WEIGHT,
) -> float:
    """
    Weighted combination of the two signal scores.

    If *llm_score* is None (e.g., M3 stub or API failure), the stylometric
    score is used alone (effectively a weight of 1.0).
    """
    if llm_score is None:
        return round(stylometric_score, 4)

    return round(llm_weight * llm_score + stylometric_weight * stylometric_score, 4)


def distance_confidence(combined: float) -> float:
    """Distance from the decision boundary, doubled: 2 * |combined - 0.5|."""
    return round(2.0 * abs(combined - 0.50), 4)


def check_disagreement(
    llm_score: float | None,
    stylometric_score: float,
    threshold: float = DISAGREEMENT_THRESHOLD,
) -> bool:
    """
    Return True if the two signals disagree by more than *threshold*.

    If the LLM score is None (stub/failure), there is no disagreement
    to measure — returns False.
    """
    if llm_score is None:
        return False
    return abs(llm_score - stylometric_score) > threshold


def determine_label(combined: float) -> str:
    """Map combined score to label using the three-tier thresholds."""
    if combined >= 0.70:
        return "human"
    if combined <= 0.30:
        return "ai"
    return "uncertain"


def score(
    llm_result: dict[str, Any] | None,
    stylometric_result: dict[str, Any],
    llm_weight_max: float = 0.70,
) -> dict:
    """
    Full confidence scoring: combine signals, compute confidence,
    apply disagreement cap, and determine the label.

    Weights are adjusted dynamically based on text length:
    under 100 words the LLM weight increases (up to *llm_weight_max*
    at ≤50 words) because stylometric metrics are unreliable on
    short texts.  The max can be overridden per content type.

    Args:
        llm_result:  Output of llm_classifier.classify() (or None if stubbed).
        stylometric_result: Output of stylometric.analyse().
        llm_weight_max: Max LLM weight for very short texts (default 0.70).

    Returns:
        {
            "label": "human" | "ai" | "uncertain",
            "confidence": float,
            "combined_score": float,
            "disagreement_detected": bool,
            "disagreement_cap_applied": bool,
        }
    """
    llm_score = llm_result["score"] if llm_result else None
    stylometric_score = stylometric_result["score"]
    text_length = stylometric_result.get("text_length")

    llm_w, sty_w = _effective_weights(text_length, llm_weight_max)
    combined = combine(llm_score, stylometric_score, llm_w, sty_w)

    raw_confidence = distance_confidence(combined)
    disagreement = check_disagreement(llm_score, stylometric_score)

    if disagreement and raw_confidence > DISAGREEMENT_CAP:
        confidence = DISAGREEMENT_CAP
        cap_applied = True
    else:
        confidence = raw_confidence
        cap_applied = False

    label = determine_label(combined)

    return {
        "label": label,
        "confidence": confidence,
        "combined_score": combined,
        "disagreement_detected": disagreement,
        "disagreement_cap_applied": cap_applied,
    }
