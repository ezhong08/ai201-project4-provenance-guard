"""
Provenance Guard — Flask application.

Multi-signal AI-content detection service.  Accepts text submissions,
runs detection signals, and returns an attribution label with a
confidence score and transparency label text.

Current scope (M5):
  - POST /submit  — both signals live, full confidence scoring,
                    transparency labels, structured audit log
  - POST /appeal  — contest a classification, updates status
  - GET  /health  — liveness check
  - GET  /log     — recent audit log entries with filtering
"""

import concurrent.futures
import hashlib
import re
import time
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import confidence_scorer
import database as db
import labels
import llm_classifier
import stylometric

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],  # only /submit has explicit rate limiting
    storage_uri="memory://",
)


@app.errorhandler(429)
def ratelimit_error(_e):
    """Return JSON for rate-limit errors (Flask-Limiter)."""
    return jsonify({
        "error": "rate_limit_exceeded",
        "message": "You have exceeded the rate limit of 10 requests per minute. "
                   "Please wait and try again.",
    }), 429


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(text: str) -> str:
    """SHA-256 hex digest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_ip(ip: str) -> str:
    """One-way hash the submitter's IP for privacy-preserving logging."""
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()


def _normalise(raw: str) -> str:
    """
    Strip framing noise without altering the prose itself.

    - Strip leading / trailing whitespace
    - Collapse runs of 3+ newlines into 2
    - Leave the text otherwise untouched
    """
    if not isinstance(raw, str):
        raise ValueError("Content must be a string")

    text = raw.strip()

    # Collapse excessive newline runs (>2) down to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def _validate(normalised: str) -> tuple[bool, str]:
    """Return (valid, error_message)."""
    if not normalised:
        return False, "Content must be a non-empty string after normalisation."

    length = len(normalised)
    if length < 50:
        return False, (
            f"Content too short: {length} characters. "
            "Minimum is 50 characters after normalisation."
        )
    if length > 10_000:
        return False, (
            f"Content too long: {length} characters. "
            "Maximum is 10,000 characters after normalisation."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Liveness check."""
    return jsonify({
        "status": "healthy",
        "timestamp": _utcnow(),
    })


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute")
def submit():
    """
    Submit text for attribution analysis.

    Request body (JSON):
        {
            "text": "The text to analyse...",
            "creator_id": "optional-user-identifier"
        }

    Returns a structured response with the detection result.
    """
    t_start = time.perf_counter()

    # --- Parse request ---
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "invalid_request", "message": "Request body must be valid JSON."}), 400

    raw_text = data.get("text", "")
    creator_id = data.get("creator_id")

    # --- Normalise ---
    try:
        normalised = _normalise(raw_text)
    except ValueError:
        return jsonify({"error": "invalid_request", "message": "Content must be a string."}), 400

    valid, msg = _validate(normalised)
    if not valid:
        return jsonify({"error": "invalid_request", "message": msg}), 400

    content_id = _hash(normalised)

    # --- Run detection signals in parallel ---
    #
    # Signal 1 (Groq LLM) and Signal 2 (Stylometric) are independent
    # and run concurrently.  If the Groq call fails, the LLM result
    # falls back to a neutral 0.50 and the error is logged.

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_llm = executor.submit(llm_classifier.classify, normalised)
        future_stylometric = executor.submit(stylometric.analyse, normalised)

        llm_result = future_llm.result()
        stylometric_result = future_stylometric.result()

    # --- Full two-signal confidence scoring ---
    scored = confidence_scorer.score(llm_result, stylometric_result)

    label = scored["label"]
    confidence = scored["confidence"]
    combined_score = scored["combined_score"]
    disagreement = scored["disagreement_detected"]

    # --- Transparency label ---
    label_result = labels.build_label(label, confidence)

    # --- Audit log ---
    elapsed_ms = round((time.perf_counter() - t_start) * 1000)

    db.insert_entry({
        "event_type": "classification",
        "content_id": content_id,
        "content_length": len(normalised),
        "creator_id": creator_id,
        "label": label,
        "confidence": confidence,
        "combined_score": combined_score,
        "signal_1_name": "llm_classifier",
        "signal_1_score": llm_result["score"],
        "signal_1_detail": llm_result,
        "signal_2_name": "stylometric",
        "signal_2_score": stylometric_result["score"],
        "signal_2_detail": stylometric_result,
        "transparency_label_variant": label_result["variant"],
        "submitter_ip_hash": _hash_ip(request.remote_addr or "unknown"),
        "processing_time_ms": elapsed_ms,
        "status": "classified",
    })

    # --- Response ---
    _attribution_map = {
        "human": "likely_human",
        "ai": "likely_ai",
        "uncertain": "uncertain",
    }

    return jsonify({
        "content_id": content_id,
        "attribution": _attribution_map.get(label, "uncertain"),
        "confidence": confidence,
        "label": label,
        "combined_score": combined_score,
        "disagreement_detected": disagreement,
        "transparency_label": label_result["text"],
        "transparency_label_variant": label_result["variant"],
        "signals": {
            "llm_classifier": llm_result,
            "stylometric": stylometric_result,
        },
        "status": "classified",
    })


@app.route("/appeal", methods=["POST"])
def appeal():
    """
    Contest a classification.

    Request body (JSON):
        {
            "content_id": "sha256hexdigest",
            "reason": "Explanation of why the classification may be wrong."
        }

    The field may also be named "creator_reasoning" (both are accepted).
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "invalid_request", "message": "Request body must be valid JSON."}), 400

    content_id = data.get("content_id", "").strip()
    if not content_id:
        return jsonify({"error": "invalid_request", "message": "content_id is required."}), 400

    reason = data.get("reason") or data.get("creator_reasoning") or ""
    if not reason.strip():
        return jsonify({"error": "invalid_request", "message": "reason is required."}), 400

    # --- Look up original classification ---
    original = db.lookup_by_content_id(content_id)
    if original is None:
        return jsonify({"error": "not_found", "message": "No classification found for the provided content_id."}), 404

    if original.get("status") == "under_review":
        return jsonify({"error": "already_appealed", "message": "This content is already under review."}), 409

    # --- Update status ---
    db.update_status(content_id, "under_review")

    # --- Log the appeal ---
    appeal_id = str(uuid.uuid4())
    db.insert_entry({
        "entry_id": appeal_id,
        "event_type": "appeal",
        "content_id": content_id,
        "linked_entry_id": original.get("entry_id"),
        "creator_reason": reason.strip(),
        "original_label": original.get("label"),
        "original_confidence": original.get("confidence"),
        "status": "under_review",
    })

    return jsonify({
        "content_id": content_id,
        "status": "under_review",
        "original_label": original.get("label"),
        "original_confidence": original.get("confidence"),
        "message": "Your appeal has been logged. A human reviewer will examine this classification.",
        "appeal_id": appeal_id,
    })


@app.route("/log", methods=["GET"])
def audit_log():
    """
    Return recent audit log entries.

    Query params:
        limit      — max entries (default 20, max 100)
        content_id — filter to a single content item (and linked appeals)
        status     — filter by status ("classified" | "under_review")
    """
    try:
        limit = int(request.args.get("limit", 20))
    except ValueError:
        limit = 20
    limit = max(1, min(limit, 100))

    content_id = request.args.get("content_id")
    status_filter = request.args.get("status")

    entries = db.get_recent_entries(limit=limit, content_id=content_id, status=status_filter)

    return jsonify({
        "entries": entries,
        "total": len(entries),
    })


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    db.init_db()
    app.run(debug=True, host="127.0.0.1", port=5000)
