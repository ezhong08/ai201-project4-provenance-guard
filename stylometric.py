"""
Stylometric Analyser — Signal 2 (Pure Python).

Computes four structural metrics on normalised text. Each metric
captures a dimension where human and AI writing statistically diverge.
Human writing is variable; AI writing is uniform. All four metrics
converge on measuring *variability*.

Metrics:
  1. Sentence length variance (coefficient of variation)
  2. Type-token ratio (vocabulary diversity, first 500 words)
  3. Punctuation density and diversity
  4. Average dependency distance (noun-verb adjacency heuristic)

Each metric returns a float in [0.0, 1.0] where 1.0 = human-like.
The four sub-scores are averaged with equal weight (0.25 each).
A short-text penalty blends the final score toward 0.50 when the
text is under 100 words (see planning.md §3b Edge Case 4).
"""

import math
import re
from typing import Tuple


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?]+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? boundaries."""
    raw = _SENTENCE_END.split(text)
    return [s.strip() for s in raw if s.strip() and len(s.split()) >= 1]


def _word_count(sentence: str) -> int:
    return len(sentence.split())


# ---------------------------------------------------------------------------
# Metric 1 — Sentence length variance (coefficient of variation)
# ---------------------------------------------------------------------------

def analyse_sentence_length_variance(text: str) -> float:
    """
    Standard deviation of sentence word-counts divided by the mean.

    Human writers vary sentence length instinctively — a long sentence
    unpacks an idea, then a short one lands it.  AI models produce
    sentences that cluster around a comfortable modal length.

    Normalisation thresholds (CV):
        <= 0.25  →  0.0  (strongly AI — very uniform)
        >= 0.70  →  1.0  (strongly human — highly variable)
    """
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return 0.50  # cannot measure variance on a single sentence

    lengths = [_word_count(s) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    if mean_len == 0:
        return 0.50

    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_len

    # Linear normalisation between [0.25, 0.70]
    return _clamp_normalise(cv, low=0.25, high=0.70)


# ---------------------------------------------------------------------------
# Metric 2 — Type-token ratio (vocabulary diversity)
# ---------------------------------------------------------------------------

def analyse_type_token_ratio(text: str) -> float:
    """
    Unique words divided by total words, computed on the first 500 words
    to keep scores comparable across submissions of different lengths.

    Human writers draw on a richer vocabulary and repeat themselves less.
    AI models have a "preferred" vocabulary of high-probability tokens
    and reuse them more frequently.

    Normalisation thresholds (TTR):
        <= 0.45  →  0.0  (strongly AI — narrow vocabulary)
        >= 0.70  →  1.0  (strongly human — diverse vocabulary)
    """
    words = text.split()
    if not words:
        return 0.50

    sample = words[:500]
    total = len(sample)
    unique = len(set(w.lower().strip(".,;:!?()[]{}'\"—-") for w in sample))
    ttr = unique / total

    return _clamp_normalise(ttr, low=0.45, high=0.70)


# ---------------------------------------------------------------------------
# Metric 3 — Punctuation density and diversity
# ---------------------------------------------------------------------------

_PUNCT_MARKS = set(".,;:!?—…\"'()[]{}")


def _count_punctuation(text: str) -> Tuple[int, int]:
    """Return (total_punct_marks, distinct_punct_types)."""
    count = 0
    types: set[str] = set()
    for ch in text:
        if ch in _PUNCT_MARKS:
            count += 1
            types.add(ch)
    return count, len(types)


def analyse_punctuation_density(text: str) -> float:
    """
    Two sub-components averaged:

      a) Punctuation marks per sentence — higher = more human variability.
      b) Distinct punctuation types used — wider palette = more human.

    AI text favours predictable comma-period patterns.  Human writers
    use a wider punctuation palette (semicolons, dashes, parentheses)
    and distribute them unevenly across sentences.

    Normalisation:
        density:  [0.3, 1.5] marks/sentence  →  [0.0, 1.0]
        diversity: [2, 6]   distinct types    →  [0.0, 1.0]
    """
    sentences = _split_sentences(text)
    if not sentences:
        return 0.50

    # Density — marks per sentence
    total_marks, distinct_types = _count_punctuation(text)
    marks_per_sentence = total_marks / len(sentences)
    density_score = _clamp_normalise(marks_per_sentence, low=0.3, high=1.5)

    # Diversity — distinct punctuation types
    diversity_score = _clamp_normalise(distinct_types, low=2.0, high=6.0)

    return 0.5 * density_score + 0.5 * diversity_score


# ---------------------------------------------------------------------------
# Metric 4 — Average dependency distance (POS-adjacency heuristic)
# ---------------------------------------------------------------------------

# Function words that don't carry noun/verb content
_FUNCTION_WORDS: set[str] = {
    # Determiners
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "each", "every", "no", "another",
    "much", "many", "few", "several", "all", "both",
    "one", "two", "three",
    # Prepositions
    "in", "on", "at", "to", "for", "with", "from", "by",
    "about", "of", "into", "onto", "upon", "within", "without",
    "through", "during", "before", "after", "above", "below",
    "between", "among", "against", "toward", "towards",
    "around", "along", "across", "behind", "beside", "beyond",
    "inside", "outside", "under", "over", "up", "down", "off",
    "near", "since", "until", "like", "as",
    # Conjunctions
    "and", "but", "or", "nor", "yet", "so", "because", "although",
    "while", "if", "when", "where", "whether", "than",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "us", "them",
    # Auxiliaries / modals
    "be", "is", "are", "am", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should", "can", "could",
    "may", "might", "must",
    # Miscellaneous
    "not", "no", "very", "just", "then", "now", "also", "too",
    "here", "there", "only", "even", "still", "already", "always",
    "never", "often", "sometimes", "usually", "really", "quite",
    "perhaps", "maybe",
}

# Common noun suffixes (heuristic for noun detection)
_NOUN_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ship",
    "ance", "ence", "er", "or", "ist", "ism",
    "ude", "tude", "age", "al", "dom",
)

# Common main-verb lemmas (contentful verbs, not auxiliaries)
_MAIN_VERBS: set[str] = {
    "say", "says", "said", "saying",
    "go", "goes", "went", "gone", "going",
    "get", "gets", "got", "getting",
    "make", "makes", "made", "making",
    "know", "knows", "knew", "knowing",
    "think", "thinks", "thought", "thinking",
    "see", "sees", "saw", "seen", "seeing",
    "come", "comes", "came", "coming",
    "take", "takes", "took", "taken", "taking",
    "give", "gives", "gave", "given", "giving",
    "find", "finds", "found", "finding",
    "tell", "tells", "told", "telling",
    "ask", "asks", "asked", "asking",
    "work", "works", "worked", "working",
    "seem", "seems", "seemed", "seeming",
    "feel", "feels", "felt", "feeling",
    "try", "tries", "tried", "trying",
    "leave", "leaves", "left", "leaving",
    "call", "calls", "called", "calling",
    "keep", "keeps", "kept", "keeping",
    "let", "lets", "letting",
    "begin", "begins", "began", "begun", "beginning",
    "show", "shows", "showed", "shown", "showing",
    "hear", "hears", "heard", "hearing",
    "play", "plays", "played", "playing",
    "run", "runs", "ran", "running",
    "move", "moves", "moved", "moving",
    "live", "lives", "lived", "living",
    "believe", "believes", "believed", "believing",
    "hold", "holds", "held", "holding",
    "bring", "brings", "brought", "bringing",
    "happen", "happens", "happened", "happening",
    "write", "writes", "wrote", "written", "writing",
    "sit", "sits", "sat", "sitting",
    "stand", "stands", "stood", "standing",
    "lose", "loses", "lost", "losing",
    "pay", "pays", "paid", "paying",
    "meet", "meets", "met", "meeting",
    "set", "sets", "setting",
    "read", "reads", "reading",
    "talk", "talks", "talked", "talking",
    "watch", "watches", "watched", "watching",
    "walk", "walks", "walked", "walking",
    "eat", "eats", "ate", "eating",
    "wait", "waits", "waited", "waiting",
    "sleep", "sleeps", "slept", "sleeping",
    "love", "loves", "loved", "loving",
    "like", "likes", "liked", "liking",
    "remember", "remembers", "remembered", "remembering",
    "forget", "forgets", "forgot", "forgotten", "forgetting",
    "look", "looks", "looked", "looking",
    "use", "uses", "used", "using",
    "help", "helps", "helped", "helping",
    "want", "wants", "wanted", "wanting",
    "need", "needs", "needed", "needing",
    "turn", "turns", "turned", "turning",
    "start", "starts", "started", "starting",
    "stop", "stops", "stopped", "stopping",
    "change", "changes", "changed", "changing",
    "speak", "speaks", "spoke", "spoken", "speaking",
    "learn", "learns", "learned", "learning",
    "understand", "understands", "understood", "understanding",
    "explain", "explains", "explained", "explaining",
    "describe", "describes", "described", "describing",
    "develop", "develops", "developed", "developing",
    "create", "creates", "created", "creating",
    "build", "builds", "built", "building",
    "grow", "grows", "grew", "grown", "growing",
    "fall", "falls", "fell", "fallen", "falling",
    "rise", "rises", "rose", "risen", "rising",
    "draw", "draws", "drew", "drawn", "drawing",
    "sing", "sings", "sang", "sung", "singing",
    "dance", "dances", "danced", "dancing",
    "smile", "smiles", "smiled", "smiling",
    "laugh", "laughs", "laughed", "laughing",
    "cry", "cries", "cried", "crying",
    "die", "dies", "died", "dying",
    "kill", "kills", "killed", "killing",
    "break", "breaks", "broke", "broken", "breaking",
    "put", "puts", "putting",
    "push", "pushes", "pushed", "pushing",
    "pull", "pulls", "pulled", "pulling",
    "open", "opens", "opened", "opening",
    "close", "closes", "closed", "closing",
    "carry", "carries", "carried", "carrying",
    "lead", "leads", "led", "leading",
    "follow", "follows", "followed", "following",
    "reach", "reaches", "reached", "reaching",
    "send", "sends", "sent", "sending",
    "receive", "receives", "received", "receiving",
    "buy", "buys", "bought", "buying",
    "sell", "sells", "sold", "selling",
    "drive", "drives", "drove", "driven", "driving",
    "fly", "flies", "flew", "flown", "flying",
    "swim", "swims", "swam", "swum", "swimming",
    "wear", "wears", "wore", "worn", "wearing",
    "teach", "teaches", "taught", "teaching",
    "serve", "serves", "served", "serving",
    "offer", "offers", "offered", "offering",
    "expect", "expects", "expected", "expecting",
    "hope", "hopes", "hoped", "hoping",
    "wish", "wishes", "wished", "wishing",
    "mean", "means", "meant", "meaning",
    "matter", "matters", "mattered", "mattering",
    "appear", "appears", "appeared", "appearing",
    "remain", "remains", "remained", "remaining",
    "continue", "continues", "continued", "continuing",
    "consider", "considers", "considered", "considering",
    "suggest", "suggests", "suggested", "suggesting",
}


def _clean_token(word: str) -> str:
    return word.lower().strip(".,;:!?()[]{}'\"—-")


def _is_content_noun(word: str, prev_word: str | None) -> bool:
    """
    Heuristic: is *word* likely a noun?

    Signals (any one is enough):
      - Follows a determiner (the cat, a dog)
      - Has a common noun suffix (development, happiness)
    """
    clean = _clean_token(word)
    if len(clean) < 2:
        return False

    # Determiner → noun pattern
    if prev_word is not None and _clean_token(prev_word) in _DETERMINERS:
        return True

    # Noun suffix pattern
    if clean.endswith(_NOUN_SUFFIXES):
        return True

    return False


def _is_content_verb(word: str) -> bool:
    """Heuristic: common main-verb lemma, or -ing/-ed suffix on a content word."""
    clean = _clean_token(word)
    if len(clean) < 2:
        return False
    if clean in _FUNCTION_WORDS:
        return False        # exclude auxiliaries / function words
    if clean in _MAIN_VERBS:
        return True
    if clean.endswith("ing") and len(clean) > 4:
        return True
    if clean.endswith("ed") and len(clean) > 3:
        return True
    return False


# Re-use _DETERMINERS from the function-words set defined above
_DETERMINERS: set[str] = {
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "each", "every", "no", "another",
    "much", "many", "few", "several", "all", "both",
    "one", "two", "three",
}


def analyse_dependency_distance(text: str) -> float:
    """
    Approximated mean word-distance between the main subject and main
    verb of each sentence via simple POS-tag adjacency heuristics
    (no full dependency parser).

    The key insight: in AI-generated text, the main subject and main
    verb tend to be adjacent or nearly adjacent (the cat sat...).
    Human writers routinely insert adjectives, relative clauses,
    prepositional phrases, and parentheticals between subject and
    verb, creating longer dependency spans.

    Algorithm per sentence:
      1. Find the first likely noun — this is the main subject.
      2. Find the first main verb that follows it.
      3. Distance = verb_position - noun_position (in words).

    Normalisation thresholds (mean word-distance):
        <= 1.5  →  0.0  (AI-like — subject-verb adjacent)
        >= 5.0  →  1.0  (human-like — substantial separation)
    """
    sentences = _split_sentences(text)
    if not sentences:
        return 0.50

    distances: list[int] = []

    for sentence in sentences:
        words = sentence.split()
        if len(words) < 4:
            continue

        # 1. Find the first noun (subject)
        subject_pos: int | None = None
        for i, word in enumerate(words):
            prev = words[i - 1] if i > 0 else None
            if _is_content_noun(word, prev):
                subject_pos = i
                break

        if subject_pos is None:
            continue

        # 2. Find the first main verb after the subject
        verb_pos: int | None = None
        for j in range(subject_pos + 1, len(words)):
            if _is_content_verb(words[j]):
                verb_pos = j
                break

        if verb_pos is None:
            continue

        # 3. Distance in words
        distances.append(verb_pos - subject_pos)

    if not distances:
        return 0.50

    mean_distance = sum(distances) / len(distances)
    return _clamp_normalise(mean_distance, low=1.5, high=5.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_normalise(value: float, low: float, high: float) -> float:
    """Map *value* from [low, high] to [0.0, 1.0], clamped at the edges."""
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def _word_count_total(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse(text: str) -> dict:
    """
    Run all four stylometric metrics on *text* (already normalised).

    Returns:
        {
            "score": float,           # average of the four sub-scores
            "weight": 0.45,           # default weight in composite scoring
            "sub_scores": {
                "sentence_length_variance": float,
                "type_token_ratio": float,
                "punctuation_density": float,
                "avg_dependency_distance": float
            },
            "text_length": int,
            "short_text_penalty_applied": bool
        }
    """
    sub_scores = {
        "sentence_length_variance": analyse_sentence_length_variance(text),
        "type_token_ratio": analyse_type_token_ratio(text),
        "punctuation_density": analyse_punctuation_density(text),
        "avg_dependency_distance": analyse_dependency_distance(text),
    }

    raw_average = sum(sub_scores.values()) / 4.0
    word_count = _word_count_total(text)
    penalty_applied = False

    # Short-text penalty: under 100 words, blend toward 0.50 (neutral).
    # The stylometric metrics are unreliable with too few observations.
    if word_count < 100:
        blend = word_count / 100.0        # 0.0 at 0 words, 1.0 at 100
        score = blend * raw_average + (1.0 - blend) * 0.50
        penalty_applied = True
    else:
        score = raw_average

    return {
        "score": round(score, 4),
        "weight": 0.45,
        "sub_scores": {k: round(v, 4) for k, v in sub_scores.items()},
        "text_length": word_count,
        "short_text_penalty_applied": penalty_applied,
    }
