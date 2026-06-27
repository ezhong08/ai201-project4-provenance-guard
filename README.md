# Provenance Guard

A multi-signal AI-content detection service that analyses submitted text and returns an attribution label with a confidence score and a plain-language transparency label. Built with Flask, Groq, and pure-Python stylometric heuristics.

---

## Architecture

```
POST /submit
  → [raw text]
  → Rate Limiter           → [raw text, allowed]
  → Input Normaliser       → [normalised text]
  → LLM Classifier (Groq)  → [llm_score: 0.90]
  → Stylometric Analyser   → [stylometric_score: 0.76, 4 sub-scores]
  → Confidence Scorer      → [combined: 0.84, confidence: 0.87, label: "human"]
  → Transparency Label     → ["Likely Human-Authored — High Confidence \n\n ..."]
  → Audit Logger (SQLite)  → [row written to audit_log table]
  → Response Builder       → [200 JSON response]
  → User
```

Two independent signals run in parallel (ThreadPoolExecutor), then converge in the confidence scorer. The full architecture diagram with both submission and appeal flows is in [planning.md §1](planning.md).

---

## Detection Signals

Provenance Guard uses two genuinely independent signals. One is semantic, one is structural. They can (and do) disagree — and when they do, that disagreement drives the combined score toward the uncertain middle.

### Signal 1 — LLM Classifier (Groq, `llama-3.3-70b-versatile`)

**What it measures:** Semantic and stylistic authenticity — whether the text reads as though a human mind with lived experience produced it, or whether it reads as a statistically plausible synthesis.

**How it works:** The text is sent to Groq's API with a structured prompt asking the model to assess: naturalness of voice, presence of personal perspective, idiosyncratic word choices, emotional texture, and whether the prose feels "lived-in" versus synthesised. The model returns a verdict (`human` / `ai` / `uncertain`) and a confidence level (`high` / `medium` / `low`), which are mapped to a numeric score on a 7-point discrete scale:

| Verdict | Confidence | Numeric score | Meaning |
|---|---|---|---|
| human | high | 0.90 | Strong semantic signal for human |
| human | medium | 0.75 | Moderate lean toward human |
| human | low | 0.60 | Weak lean — could be uncertain |
| uncertain | any | 0.50 | Model cannot decide — neutral |
| ai | low | 0.40 | Weak lean toward AI |
| ai | medium | 0.25 | Moderate lean toward AI |
| ai | high | 0.10 | Strong semantic signal for AI |

**Why this signal:** LLMs can detect the "smooth but hollow" quality of AI prose — the lack of a genuine human perspective behind the words — which is invisible to structural heuristics. And by using a model from the same family as many AI content generators, the classifier is most sensitive to the statistical fingerprints it's best positioned to detect.

**Blind spots:** Highly stylised human writing (experimental fiction, stream-of-consciousness) can read as incoherent and be flagged as AI. Non-native English prose — with simpler syntax and more limited vocabulary — may be misread as AI-generated. Prompt injection risk: if the submitted text itself contains LLM-formatted instructions, the classifier prompt must insulate judgment from content (our prompt uses a `---` sandwich to isolate the text being analysed).

### Signal 2 — Stylometric Analyser (Pure Python)

**What it measures:** Four structural properties that statistically diverge between human and AI writing. All four converge on a single underlying property: *variability*. Human writing is variable — in sentence length, vocabulary, punctuation, and syntactic complexity — because humans think variably. AI writing is optimised for fluency, and fluency means smoothing out variance.

| Metric | Human tendency | AI tendency |
|---|---|---|
| **Sentence length variance** (CV) | High — mix of long and short sentences | Low — sentences cluster near a modal length |
| **Type-token ratio** (TTR) | Higher — richer vocabulary, less repetition | Lower — reuses a smaller working vocabulary |
| **Punctuation density & diversity** | Variable — semicolons, dashes, fragments | Uniform — predictable comma-period patterns |
| **Dependency distance** (subject–verb) | Longer — embedded clauses, deferred resolution | Shorter — linear, right-branching structures |

Each metric is normalised to `[0.0, 1.0]` using fixed calibration thresholds, then averaged with equal weight. Texts under 100 words trigger a short-text penalty that blends the stylometric score toward 0.50 (neutral), since the metrics are unreliable with too few observations.

**Why these two signals together:** One reads the text holistically for voice and authenticity; the other measures structural variance with a ruler. An adversary who defeats one signal has not automatically defeated the other. The stylometric signal costs nothing (no API call), runs in <1ms, and provides a cross-check against the LLM classifier's architectural blind spots.

---

## Confidence Scoring

The confidence scorer combines both signals and produces three outputs: a combined score, a confidence value, and a label.

### Algorithm

1. **Weighted combination:** `combined = 0.55 × llm_score + 0.45 × stylometric_score`
   - The LLM gets a slight edge (0.55 vs 0.45) because it captures dimensions that structural metrics miss entirely — but the weights are close to equal so neither signal can dominate alone.
   - For texts under 100 words, the LLM weight increases linearly from 0.55 to 0.70 at 50 words, since stylometric metrics are unreliable on short texts ([planning.md §3b Edge Case 4](planning.md)).

2. **Distance-from-boundary confidence:** `confidence = 2 × |combined - 0.5|`
   - A 0.51 combined score → confidence 0.02 (barely leaning). A 0.95 combined → confidence 0.90 (strongly certain). The score tells you how *strongly* the signals converge, not just which side they land on.

3. **Signal disagreement cap:** If `|llm_score - stylometric_score| > 0.40`, confidence is capped at 0.60 regardless of the combined score. Conflicting signals mean the system should not speak with certainty.

4. **Label thresholds:**

| Combined Score | Label | Meaning |
|---|---|---|
| ≥ 0.70 | `human` | Both signals lean strongly toward human authorship |
| ≤ 0.30 | `ai` | Both signals lean strongly toward AI generation |
| 0.31 – 0.69 | `uncertain` | Signals are conflicted, weak, or near the boundary |

### Why this approach

A binary classifier at 0.5 would hide all the interesting information. The system needs to distinguish between "both signals barely lean AI at 0.49" and "both signals emphatically point AI at 0.05" — these should produce different labels for the human reader. The distance-from-boundary formula captures this: the further from 0.5, the stronger the agreement. The disagreement cap prevents the system from sounding certain when its own signals contradict each other.

### Example submissions — meaningful score variation

**High-confidence case — clearly AI-generated text (115 words):**

```
Input: "Technology has changed the way people communicate in the modern world.
Technology has made communication faster and more efficient for many people..."
(repetitive, uniform sentence structure, no personal voice)

LLM Classifier:    0.10  (ai, high confidence)
                    ↳ "The text lacks a personal perspective, uses repetitive
                       and generic phrases, and has a synthesized tone."

Stylometric:       0.19  (strongly AI-like)
  sentence_length_variance: 0.00  (all sentences same length)
  type_token_ratio:         0.04  (extreme vocabulary repetition)
  punctuation_density:      0.29  (only commas and periods)
  dependency_distance:      0.43  (short subject–verb spans)

→ combined = 0.14   →   label = ai   →   confidence = 0.72
```

**Lower-confidence case — formal academic writing (43 words):**

```
Input: "The relationship between monetary policy and asset price inflation has
been extensively studied in the literature. Central banks face a fundamental
tension between their mandate for price stability..."

LLM Classifier:    0.25  (ai, medium confidence)
                    ↳ "The text's formal tone and lack of personal
                       perspective suggest AI generation, though its
                       coherence makes it plausible human writing."

Stylometric:       0.53  (near neutral — short text, penalty applied)
  sentence_length_variance: 0.01  (uniform academic sentences)
  type_token_ratio:         1.00  (all words unique in 43-word sample)
  punctuation_density:      0.29
  dependency_distance:      1.00

→ combined = 0.33   →   label = uncertain   →   confidence = 0.33
```

The first case produces confidence 0.72 — the system is fairly sure. The second produces 0.33 — the system admits it doesn't know. **These are meaningfully different outputs from meaningfully different inputs.** The formal academic text lands in the uncertain middle because the LLM misreads its impersonal register as AI-like (a known blind spot) and the short text makes the stylometric signal unreliable.

---

## Transparency Labels

The transparency label is what a non-technical reader actually sees. It is **gated on confidence, not on the binary label** — a 0.51 classification and a 0.95 classification produce different user-facing text even if both have the same API label. Confidence below 0.50 always produces Label C regardless of the binary result.

### Label A — High-confidence AI (`combined ≤ 0.30`, `confidence ≥ 0.50`)

> **AI-Generated Content — High Confidence**
>
> Our detection system is strongly confident that this text was generated by artificial intelligence. Two independent signals — a semantic analysis by an AI model and a statistical analysis of the writing's structure — both point toward AI authorship. AI-generated content can be useful, but it may contain errors or lack original insight. Please read critically.

### Label B — High-confidence Human (`combined ≥ 0.70`, `confidence ≥ 0.50`)

> **Likely Human-Authored — High Confidence**
>
> Our detection system is strongly confident that this text was written by a human. It shows the natural variability in sentence structure, vocabulary diversity, and authentic voice that characterise human writing. As with any attribution system, this is an informed estimate — not a guarantee.

### Label C — Uncertain (all other cases)

> **Attribution Uncertain**
>
> Our detection system could not reach a confident conclusion about whether this text was written by a human or generated by AI. Our two detection signals — one analysing writing style semantically and one measuring structural patterns — produced mixed or borderline results. This is common with short texts, highly polished writing, creative prompts designed to mimic human style, or non-native English prose. Treat this content as unverified.

### Why the uncertain label is the safety net

Roughly the middle 50% of the score range produces Label C. The system is deliberately conservative about claiming it knows. A false positive — publicly branding a human author's work as AI-generated — is worse than a false negative on a writing platform. The uncertain label errs on the side of admitting doubt, and the text itself explains *why* the system might be struggling (short text, polished style, non-native English) so the reader has context.

---

## Appeals Workflow

Creators can contest a classification via `POST /appeal`. The workflow:

1. Looks up the original classification by `content_id` → 404 if not found
2. Checks whether the content is already under review → 409 if so
3. Updates the content's status from `classified` to `under_review` in the audit log
4. Inserts a new audit log entry with `event_type: "appeal"`, linking to the original classification
5. Returns a confirmation with an `appeal_id`

The appeal does **not** automatically re-classify — it flags the content for human review. The appeal and the original classification are linked in the audit log via `linked_entry_id`, so a reviewer can see the full decision history.

```
POST /appeal
Request:  {"content_id": "abc123...", "reason": "I wrote this myself..."}
Response: {"content_id": "abc123...", "status": "under_review",
           "original_label": "ai", "original_confidence": 0.33,
           "appeal_id": "uuid-v4",
           "message": "Your appeal has been logged..."}
```

---

## Rate Limiting

| Setting | Value | Reasoning |
|---|---|---|
| Library | Flask-Limiter (in-memory) | Lightweight, no external Redis dependency needed for a proof-of-concept |
| Limit | 10 requests per minute per IP | Each `/submit` triggers one Groq API call. At 10 req/min, worst case is ~14,400 calls/day — within Groq's free tier while leaving headroom. Prevents a single user from exhausting the API budget |
| Window | Sliding (1 minute) | Short enough to catch abuse bursts, long enough to allow legitimate sequential use |
| Scope | Per IP address | Simplest fair allocation for a proof-of-concept |
| Response | `429` with JSON error body | Machine-readable so clients can implement backoff |

---

## Audit Log

Every attribution decision and appeal is captured in a structured SQLite database (`audit_log.db`). The schema has 24 columns covering: entry ID (UUID v4), timestamp (ISO-8601 UTC), event type (`classification` or `appeal`), content ID (SHA-256 of normalised text), content length, creator ID, label, confidence, combined score, both individual signal scores with full detail JSON, transparency label variant, submitter IP hash, processing time in ms, status (`classified` or `under_review`), and appeal-specific fields (`linked_entry_id`, `creator_reason`, `original_label`, `original_confidence`).

The log is queryable via `GET /log` with optional `?limit=N`, `?content_id=<hash>`, and `?status=classified|under_review` filters.

Example entry:

```json
{
  "entry_id": "a1b2c3d4-...",
  "timestamp": "2026-06-26T21:35:00Z",
  "event_type": "classification",
  "content_id": "26e36197f3b1...",
  "label": "ai",
  "confidence": 0.68,
  "combined_score": 0.14,
  "signal_1_score": 0.10,
  "signal_2_score": 0.19,
  "transparency_label_variant": "A",
  "status": "classified"
}
```

---

## Known Limitations

### Formal professional writing is structurally indistinguishable from AI output

A journalist writing to the Associated Press style guide or an academic writing for a journal produces prose that is intentionally impersonal, with controlled sentence complexity, constrained vocabulary (grade 8–10 reading level), and no semicolons, em-dashes, or parentheticals (AP style forbids them). **Both signals fail on this content:**

- **Signal 1 (LLM Classifier):** The journalist's voice is suppressed by professional convention. No idiosyncrasy, no digression, no emotional texture. This is exactly the "hollow fluency" pattern the LLM flags as AI. The model may also have a prior toward AI for this genre, knowing that many news outlets use AI to generate routine articles.
- **Signal 2 (Stylometric):** AP style enforces exactly the uniformity the stylometric analyser measures as AI-like. Sentence length variance is moderate but controlled (style guides discourage both very long and very short sentences). Punctuation diversity is intentionally low. On all four metrics, the article looks like well-edited AI output — because both AP style and LLM training converge on the same "clear, accessible prose" target.

In our testing, a paragraph of academic economics prose scored `combined = 0.33` with `confidence = 0.33` — uncertain, but leaning AI. The system correctly hedges (Label C), but the binary label was `"ai"` for content that was human-written. **This is the system's worst-case failure mode:** confidently wrong would be inexcusable; uncertain-but-wrong is still wrong, just honestly so. A production deployment would need a third signal calibrated specifically for formal registers, or genre-specific thresholds that relax the AI-leaning interpretation when the text matches known formal-writing patterns.

### Additional blind spots

- **Short texts (<100 words):** The stylometric signal is dominated by noise at short lengths. The LLM weight increases for short texts, but this means the system is essentially single-signal below 50 words.
- **Non-native English:** Simpler syntax and limited vocabulary may be misread as AI-generated by both signals.
- **Adversarial editing:** A determined user can manually edit AI output to inject variance (break up sentences, swap synonyms, add punctuation) — raising the stylometric score with ~5 minutes of effort.

---

## Spec Reflection

### How the spec guided implementation

The architecture diagram in [planning.md §1](planning.md) made the parallel signal execution obvious before a single line of code was written. The diagram shows Signal 1 and Signal 2 as independent branches that converge at the Confidence Scorer — this translated directly into `ThreadPoolExecutor` with two futures. Without the diagram, I might have written sequential calls and only later realised the Groq API latency (~500ms) was doubling response time for no reason. The spec forced me to think about data flow before control flow.

### How the implementation diverged from the spec

The spec defined the confidence gate for transparency labels at **0.80** — meaning confidence below 0.80 always produced Label C (Uncertain). The intent was sound: only show a confident label when the system is genuinely sure. But once both signals were calibrated and running against real text, it became clear that 0.80 was unreachable. To hit confidence ≥ 0.80, the combined score needs to be ≤ 0.10 or ≥ 0.90 — which requires both signals to agree at near-maximum strength. In practice, the LLM classifier caps at 0.90 (the "human + high confidence" mapping) and the stylometric analyser caps around 0.65 for even the most varied human prose (TTR and dependency distance have natural ceilings). Combined: `0.55 × 0.90 + 0.45 × 0.65 = 0.788` → confidence 0.58.

The implementation uses **0.50** as the gate. At this threshold, combined ≥ 0.75 or ≤ 0.25 triggers the confident labels — achievable with real signal ranges while still requiring both signals to lean the same direction. The trade-off: more content gets Label A or B instead of C. The safeguard: Label C still covers the broad middle where signals are weak or conflicted (~50% of the score range). If I were deploying this for real, I'd recalibrate the stylometric thresholds against a reference corpus so higher scores were achievable, then raise the gate back toward 0.60–0.70.

---

## AI Usage

### Instance 1 — Stylometric analyser (M3)

**What I directed the AI to do:** Given the detection signals spec from planning.md §2 (the four-metric table, normalisation strategy, and blind spots) and the architecture diagram, generate the `stylometric.py` module.

**What it produced:** A working four-metric analyser with sentence length variance, type-token ratio, punctuation density, and dependency distance functions.

**What I revised:** The dependency distance metric was the most heavily revised. The initial implementation used an all-pairs noun-verb distance heuristic that produced noisy, sometimes inverted results (AI text was scoring higher than human text because the heuristic was measuring the wrong thing). I rewrote it to measure *main subject to main verb* distance per sentence — the distance between the first noun and the first verb that follows it. This captures the linguistic insight (AI text keeps subject and verb adjacent; human text inserts words between them) much more reliably. I also tuned all four normalisation thresholds against real human and AI text samples until the separation was ≥0.30 between clearly human and clearly AI inputs.

### Instance 2 — Confidence scoring gate calibration (M4/M5)

**What I directed the AI to do:** Given the confidence scoring spec from planning.md §1 Step 5 (weighted combination, distance-from-boundary, disagreement cap, label thresholds), generate the `confidence_scorer.py` module.

**What it produced:** A correct implementation of the weighted combination formula, the `2 × |combined - 0.5|` confidence calculation, the 0.40 disagreement threshold, and the three-tier label thresholds.

**What I revised:** Two things. First, the short-text weight adjustment — the initial design gave the LLM 0.80 weight for texts ≤50 words, which let a single fooled LLM call override the stylometric entirely (an adversarially-edited text scored `combined = 0.71` — human — when it should have been uncertain). I dropped the max from 0.80 to 0.70 so the stylometric retains meaningful influence even on short texts. Second, the transparency label confidence gate — the spec's 0.80 was unreachable with real signal ranges, so I calibrated it to 0.50 after testing against all four inputs across the confidence spectrum. The `labels.py` module documents this calibration decision.

---

## API Reference

### `POST /submit`

Submit text for attribution analysis. Rate limited: 10 requests/minute/IP.

**Request:**
```json
{
  "text": "The text to analyse (50–10,000 characters after normalisation).",
  "creator_id": "optional-user-identifier"
}
```

**Response `200 OK`:**
```json
{
  "content_id": "sha256hexdigest",
  "attribution": "likely_ai",
  "label": "ai",
  "confidence": 0.72,
  "combined_score": 0.14,
  "disagreement_detected": false,
  "transparency_label": "AI-Generated Content — High Confidence\n\n...",
  "transparency_label_variant": "A",
  "signals": {
    "llm_classifier": { "score": 0.10, "raw_verdict": "ai", "raw_confidence": "high", "weight": 0.55, "model": "llama-3.3-70b-versatile" },
    "stylometric": { "score": 0.19, "weight": 0.45, "sub_scores": { "sentence_length_variance": 0.00, "type_token_ratio": 0.04, "punctuation_density": 0.29, "avg_dependency_distance": 0.43 } }
  },
  "status": "classified"
}
```

### `POST /appeal`

Contest a classification.

**Request:** `{"content_id": "sha256hexdigest", "reason": "..."}`  
**Response:** `{"content_id": "...", "status": "under_review", "original_label": "ai", "original_confidence": 0.33, "appeal_id": "uuid-v4", "message": "..."}`  
**Errors:** `404` (not found), `409` (already appealed)

### `GET /log`

Return audit log entries.

**Query params:** `?limit=N` (default 20, max 100), `?content_id=<hash>`, `?status=classified|under_review`

### `GET /health`

Liveness check: `{"status": "healthy", "timestamp": "..."}`

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env         # add your GROQ_API_KEY
python app.py                # starts on http://127.0.0.1:5000
```
