"""Multi-pass voting consensus (ROVER) over Whisper hypotheses.

The problem this solves: a single Whisper pass sometimes invents a line over a
held note and sometimes swallows one at a window edge.  Both failures are
*unstable* — change the decoding temperature or move the window boundary and
they move too.  Real lyrics do not move.

So we run the same audio several times with deliberately decorrelated settings,
align the hypotheses word by word into a confusion network, and let each slot
vote.  A word that only one pass heard loses to the silence the others heard;
a word every pass heard survives even when they spell it differently.  The
iteration stops when adding a pass stops changing the answer, which is the
convergence check the whole design is built around.

Pure functions over plain data: no audio, no models, no I/O.
"""

from __future__ import annotations

# Interfaz externa de este módulo; lo que no está aquí es seam interno
# y solo lo tocan sus propias pruebas (docs/arquitectura.md).
__all__ = [
    "ConsensusStats", "ConsensusWord", "consensus_words", "is_filler_line",
    "single_pass_words", "summarize", "to_segments",
]

import math
import re
import statistics
import unicodedata
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass, field

from textjoin import join_tokens, space_prefixes

# Voting weights.  `VOTE_WEIGHT` is how much raw agreement counts against the
# acoustic confidence the engine reported.  Agreement dominates on purpose:
# confidence is exactly the number Whisper gets wrong when it hallucinates.
VOTE_WEIGHT = 0.75
CONFIDENCE_WEIGHT = 1.0 - VOTE_WEIGHT

# Score the "this word was not said" hypothesis gets per abstaining pass.
# Below 0.5 the network keeps too much invented text; above 0.7 it starts
# deleting quiet real words.
NULL_CONFIDENCE = 0.55

# Alignment: two words can only be the same word if they were heard at roughly
# the same moment.  A chorus repeats identical text minutes apart, and without
# this guard the aligner happily matches those.
MAX_ALIGN_DRIFT = 2.5
GAP_PENALTY = -1.2
MATCH_SCORE = 2.0
SIMILAR_SCORE = 0.7
MISMATCH_SCORE = -1.4
SIMILAR_RATIO = 0.74

# Re-segmentation: where the consensus word stream becomes lines again.
SEGMENT_PAUSE = 0.8
# Subtitle-sized lines are easier to read and edit than Whisper's original
# paragraphs.  Punctuation is preferred, while these hard ceilings still work
# when the model returns an unpunctuated stream.
SEGMENT_MAX_SECONDS = 8.0
SEGMENT_MAX_WORDS = 14
SEGMENT_MAX_CHARS = 84
SEGMENT_MIN_CLAUSE_WORDS = 3
SEGMENT_MIN_CLAUSE_SECONDS = 1.25

# Repetition-loop guard.  Whisper sometimes latches onto a phrase and repeats it
# until the window ends; when every pass latches the same way, voting cannot see
# it — all the passes agree, they are just all wrong together.  Songs do repeat,
# so the guard only fires on runs far longer than any real hook, and it keeps
# the first few repetitions rather than deleting the phrase outright.
REPEAT_NGRAM_MAX = 6
REPEAT_LIMIT = 8
REPEAT_KEEP = 4

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)
_STRONG_END = re.compile(r"[.!?…。！？][\"'’”)\]}]*$")
_CLAUSE_END = re.compile(r"[,;:，、；：][\"'’”)\]}]*$")

# Whisper learned these from subtitled video and emits them over music and
# silence.  Voting cannot remove them — every pass hallucinates the same
# caption, so they are unanimous — and they are never song lyrics.  Matched
# against a whole line only: "thank you" inside a real verse is left alone.
_FILLER_PHRASES = frozenset({
    # ja — subtitle sign-offs, by far the most common on music
    "ご視聴ありがとうございました", "ご視聴ありがとうございます",
    "字幕をご視聴頂きましてありがとうございました",
    "最後までご視聴頂きありがとうございます",
    "チャンネル登録お願いします", "おやすみなさい",
    "エンディング", "本編はコチラ",
    # Lo que Whisper-ja escribe cuando se le fuerza el japonés sobre música sin
    # voz: medido sobre el stem de 1b0734de70f7, nunca como letra.
    "お疲れ様でした", "音楽",
    # en
    "thank you", "thanks for watching", "thank you for watching",
    "please subscribe", "subscribe to my channel",
    "thank you very much", "bye",
    # Rótulos de sección: lo que sale de la mezcla en un puente instrumental.
    "outro", "outro music", "music", "bgm",
    # ko — despedida de vídeo; sale aunque la pista no tenga nada en coreano.
    "다음 주에 만나요", "다음주에 만나요",
    # es / pt
    "gracias por ver el video", "suscríbete al canal",
    "obrigado por assistir",
})


# Sin espacios: Whisper parte el japonés en tokens ("ご 視 聴 ありがとう…") y
# el coreano los pone donde quiere, así que la comparación no puede contar con
# ellos.
_FILLER_KEYS = frozenset(re.sub(r"\s+", "", phrase) for phrase in _FILLER_PHRASES)
# Los créditos que Whisper inventa al cierre de un vídeo ("作詞・作曲・編曲
# 初音ミク"): una letra cantada no empieza nunca así.
_CREDIT_LINE = re.compile(r"^(作詞|作曲|編曲)")


def is_filler_line(text: str) -> bool:
    """Is this whole line one of Whisper's caption reflexes?"""
    stripped = re.sub(r"[\s.,!?…。、！？]+$", "", text.strip())
    key = re.sub(r"\s+", "", unicodedata.normalize("NFKC", stripped).casefold())
    return key in _FILLER_KEYS or bool(_CREDIT_LINE.match(key))


@dataclass
class ConsensusWord:
    text: str
    start: float
    end: float
    votes: int
    passes: int
    confidence: float
    agreement: float
    variants: list[str] = field(default_factory=list)

    @property
    def contested(self) -> bool:
        """True when the passes did not agree unanimously on this word."""
        return self.votes < self.passes or len(self.variants) > 1


@dataclass
class ConsensusStats:
    """What was removed, and by which of the two mechanisms.

    Kept apart because they mean opposite things about the passes: `dropped`
    counts words the passes *disagreed* about and silence won, while `looped`
    counts words they all agreed on and were wrong about together.  A single
    combined number would let a big loop masquerade as healthy disagreement.
    """

    dropped: int = 0
    looped: int = 0

    @property
    def removed(self) -> int:
        return self.dropped + self.looped


@dataclass
class RoundReport:
    """One iteration of the convergence loop, as shown in the UI."""

    passes: int
    words: int
    dropped: int
    looped: int
    mean_confidence: float
    mean_agreement: float
    unanimous: float
    # 1.0 means this round produced exactly the same text as the previous one.
    stability: float | None = None

    def as_dict(self) -> dict:
        return {
            "passes": self.passes,
            "words": self.words,
            "dropped": self.dropped,
            "looped": self.looped,
            "meanConfidence": round(self.mean_confidence, 4),
            "meanAgreement": round(self.mean_agreement, 4),
            "unanimous": round(self.unanimous, 4),
            "stability": None if self.stability is None else round(self.stability, 4),
        }


# --------------------------------------------------------------------------- #
# Normalización
# --------------------------------------------------------------------------- #
def normalize_token(text: str) -> str:
    """The form two passes must share to count as the same word."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = folded.replace("’", "'").replace("`", "'")
    return _PUNCT.sub("", folded).strip()


def join_words(tokens: list[str]) -> str:
    """Join consensus tokens; spacing is decided per boundary (see textjoin)."""
    return join_tokens(tokens)


def _similarity(a: str, b: str) -> float:
    """Cheap character-bigram Dice coefficient; good enough to catch
    "bailando" vs "bailamos" without pulling in difflib's quadratic path."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) == 1 or len(b) == 1:
        return 1.0 if a == b else 0.0
    pairs_a = {a[i:i + 2] for i in range(len(a) - 1)}
    pairs_b = {b[i:i + 2] for i in range(len(b) - 1)}
    return 2 * len(pairs_a & pairs_b) / (len(pairs_a) + len(pairs_b))


# --------------------------------------------------------------------------- #
# Red de confusión
# --------------------------------------------------------------------------- #
@dataclass
class _Slot:
    """One position in the confusion network: what each pass heard here."""

    entries: list[dict] = field(default_factory=list)
    voted: set[int] = field(default_factory=set)

    @property
    def time(self) -> float:
        return statistics.median([e["start"] for e in self.entries])

    @property
    def key(self) -> str:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry["key"]] = counts.get(entry["key"], 0) + 1
        return max(counts, key=lambda k: (counts[k], k))


def _pair_score(key_a: str, time_a: float, key_b: str, time_b: float) -> float:
    drift = abs(time_a - time_b)
    if drift > MAX_ALIGN_DRIFT:
        return MISMATCH_SCORE * 2
    penalty = drift / MAX_ALIGN_DRIFT
    if key_a == key_b:
        return MATCH_SCORE - penalty
    ratio = _similarity(key_a, key_b)
    if ratio >= SIMILAR_RATIO:
        return SIMILAR_SCORE - penalty
    return MISMATCH_SCORE - penalty


def _align(slots: list[_Slot], words: list[dict]) -> list[tuple[int | None, int | None]]:
    """Needleman–Wunsch between the network and one new hypothesis.

    Global alignment rather than difflib's longest-matching-block heuristic:
    the sequences are short (one speech region), and a proper DP is the only
    way a substitution stays a substitution instead of becoming a delete plus
    an insert in two different slots.
    """
    n, m = len(slots), len(words)
    if n == 0:
        return [(None, j) for j in range(m)]
    if m == 0:
        return [(i, None) for i in range(n)]

    slot_keys = [slot.key for slot in slots]
    slot_times = [slot.time for slot in slots]

    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + GAP_PENALTY
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + GAP_PENALTY

    for i in range(1, n + 1):
        row, prev = score[i], score[i - 1]
        key_i, time_i = slot_keys[i - 1], slot_times[i - 1]
        for j in range(1, m + 1):
            word = words[j - 1]
            diagonal = prev[j - 1] + _pair_score(
                key_i, time_i, word["key"], word["start"])
            row[j] = max(diagonal, prev[j] + GAP_PENALTY, row[j - 1] + GAP_PENALTY)

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            word = words[j - 1]
            diagonal = score[i - 1][j - 1] + _pair_score(
                slot_keys[i - 1], slot_times[i - 1], word["key"], word["start"])
            if math.isclose(score[i][j], diagonal, rel_tol=1e-9, abs_tol=1e-9):
                pairs.append((i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and math.isclose(score[i][j], score[i - 1][j] + GAP_PENALTY,
                                  rel_tol=1e-9, abs_tol=1e-9):
            pairs.append((i - 1, None))
            i -= 1
            continue
        pairs.append((None, j - 1))
        j -= 1
    pairs.reverse()
    return pairs


def build_network(hypotheses: list[list[dict]]) -> list[_Slot]:
    """Progressively align every pass into one confusion network.

    Seeded with the longest hypothesis: starting from a short one would force
    every extra word of every other pass to be an insertion, and insertions are
    where alignment quality goes to die.
    """
    order = sorted(range(len(hypotheses)), key=lambda i: -len(hypotheses[i]))
    slots: list[_Slot] = []
    for position, index in enumerate(order):
        words = hypotheses[index]
        if position == 0:
            slots = [_Slot(entries=[dict(word)], voted={word["pass"]})
                     for word in words]
            continue
        merged: list[_Slot] = []
        for slot_index, word_index in _align(slots, words):
            if slot_index is not None and word_index is not None:
                slot = slots[slot_index]
                slot.entries.append(dict(words[word_index]))
                slot.voted.add(words[word_index]["pass"])
                merged.append(slot)
            elif slot_index is not None:
                merged.append(slots[slot_index])
            else:
                word = words[word_index]
                merged.append(_Slot(entries=[dict(word)], voted={word["pass"]}))
        slots = merged
    return slots


def _resolve_slot(slot: _Slot, passes: int) -> ConsensusWord | None:
    """Vote one slot.  Returns None when silence won."""
    by_key: dict[str, list[dict]] = {}
    for entry in slot.entries:
        by_key.setdefault(entry["key"], []).append(entry)

    voters = len({entry["pass"] for entry in slot.entries})
    null_votes = max(0, passes - voters)
    null_score = VOTE_WEIGHT * (null_votes / passes) + CONFIDENCE_WEIGHT * (
        NULL_CONFIDENCE * (null_votes / passes))

    def rank(item: tuple[str, list[dict]]) -> tuple[float, int, int]:
        _key, entries = item
        votes = len({entry["pass"] for entry in entries})
        confidence = sum(entry["probability"] for entry in entries) / len(entries)
        score = (VOTE_WEIGHT * (votes / passes)
                 + CONFIDENCE_WEIGHT * confidence * (votes / passes))
        # Ties go to the variant heard by the earliest (strongest) pass.
        return score, votes, -min(entry["pass"] for entry in entries)

    best_key, best_entries = max(by_key.items(), key=rank)
    best_score = rank((best_key, best_entries))[0]
    if not best_key or best_score < null_score:
        return None

    votes = len({entry["pass"] for entry in best_entries})
    # Surface form: the exact spelling/casing/punctuation most passes wrote.
    surfaces: dict[str, int] = {}
    for entry in best_entries:
        surfaces[entry["text"]] = surfaces.get(entry["text"], 0) + 1
    surface = max(surfaces, key=lambda s: (surfaces[s], -len(s)))

    return ConsensusWord(
        text=surface,
        start=statistics.median([e["start"] for e in best_entries]),
        end=statistics.median([e["end"] for e in best_entries]),
        votes=votes,
        passes=passes,
        confidence=sum(e["probability"] for e in best_entries) / len(best_entries),
        agreement=votes / passes,
        variants=sorted(by_key.keys()),
    )


def collapse_repetition_loops(words: list[ConsensusWord]
                              ) -> tuple[list[ConsensusWord], int]:
    """Trim runaway repetitions of the same phrase.

    Scans for the *shortest* n-gram that repeats back to back and, past
    ``REPEAT_LIMIT`` repetitions, keeps only the first ``REPEAT_KEEP``.  The
    shortest period is the real repeating unit: taking a longer one would fold
    two repetitions into one "unit" and leave twice as much of the loop behind.
    """
    keys = [normalize_token(word.text) for word in words]
    keep = [True] * len(words)
    removed = 0

    index = 0
    while index < len(words):
        best_size = 0
        best_repeats = 0
        for size in range(1, min(REPEAT_NGRAM_MAX, (len(words) - index) // 2) + 1):
            pattern = keys[index:index + size]
            if not any(pattern):
                continue
            repeats = 1
            cursor = index + size
            while (cursor + size <= len(words)
                   and keys[cursor:cursor + size] == pattern):
                repeats += 1
                cursor += size
            if repeats > REPEAT_LIMIT:
                best_size, best_repeats = size, repeats
                break
        if not best_size:
            index += 1
            continue
        kept = index + best_size * REPEAT_KEEP
        end = index + best_size * best_repeats
        for position in range(kept, end):
            keep[position] = False
            removed += 1
        index = end

    if not removed:
        return words, 0
    return [word for word, alive in zip(words, keep) if alive], removed


def _repair_monotonicity(words: list[ConsensusWord]) -> list[ConsensusWord]:
    """Medians from different passes can cross; timelines cannot.

    A clip whose start precedes the previous clip's end renders as two lyrics
    on screen at once, so overlaps are squeezed rather than tolerated.
    """
    out: list[ConsensusWord] = []
    for word in words:
        start, end = word.start, max(word.end, word.start + 0.02)
        if out and start < out[-1].end:
            midpoint = (out[-1].end + start) / 2
            if midpoint > out[-1].start + 0.02:
                out[-1].end = midpoint
                start = midpoint
            else:
                start = out[-1].end
            end = max(end, start + 0.02)
        out.append(ConsensusWord(
            text=word.text, start=start, end=end, votes=word.votes,
            passes=word.passes, confidence=word.confidence,
            agreement=word.agreement, variants=word.variants,
        ))
    return out


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def consensus_words(passes: list[list[dict]], *,
                    region_bounds: list[tuple[float, float]] | None = None,
                    ) -> tuple[list[ConsensusWord], ConsensusStats]:
    """Vote a set of passes into one word stream.

    Each pass is a list of ``{"text", "start", "end", "probability"}`` covering
    the whole timeline.  Alignment happens region by region so a three-hour mix
    never builds one enormous DP matrix.

    Returns the surviving words and what was removed on the way.
    """
    count = len(passes)
    if count == 0:
        return [], ConsensusStats()

    prepared: list[list[dict]] = []
    for index, words in enumerate(passes):
        prepared.append([
            {"text": word["text"], "key": normalize_token(word["text"]),
             "start": float(word["start"]), "end": float(word["end"]),
             "probability": float(word.get("probability", 1.0)), "pass": index}
            for word in words if str(word.get("text", "")).strip()
        ])

    if region_bounds is None:
        buckets = [prepared]
    else:
        # Words arrive sorted by time, so each region's slice is found by
        # binary search.  Re-scanning every pass for every region turns a
        # three-hour mix into tens of millions of comparisons.
        starts = [[word["start"] for word in words] for words in prepared]
        buckets = []
        for start, end in region_bounds:
            buckets.append([
                words[bisect_left(times, start - 0.5):bisect_left(times, end + 0.5)]
                for words, times in zip(prepared, starts)
            ])

    resolved: list[ConsensusWord] = []
    stats = ConsensusStats()
    for bucket in buckets:
        if not any(bucket):
            continue
        for slot in build_network(bucket):
            word = _resolve_slot(slot, count)
            if word is None:
                stats.dropped += 1
            else:
                resolved.append(word)

    resolved.sort(key=lambda w: (w.start, w.end))
    resolved, stats.looped = collapse_repetition_loops(resolved)
    return _repair_monotonicity(resolved), stats


def single_pass_words(words: list[dict], passes: int = 1
                      ) -> list[ConsensusWord]:
    """Preserve the deterministic pass when multi-pass alignment collapses.

    The low agreement is intentional: the UI will flag these lines for human
    review, but usable uncertain text is safer than falsely claiming the
    entire vocal mix is instrumental.
    """
    total = max(1, passes)
    recovered = [
        ConsensusWord(
            text=str(word.get("text") or "").strip(),
            start=float(word.get("start") or 0.0),
            end=max(float(word.get("end") or 0.0),
                    float(word.get("start") or 0.0) + 0.02),
            votes=1, passes=total,
            confidence=float(word.get("probability", 1.0)),
            agreement=1.0 / total,
            variants=[normalize_token(str(word.get("text") or ""))],
        )
        for word in words if str(word.get("text") or "").strip()
    ]
    recovered, _removed = collapse_repetition_loops(recovered)
    return _repair_monotonicity(recovered)


def summarize(words: list[ConsensusWord], stats: ConsensusStats,
              passes: int, previous: list[ConsensusWord] | None) -> RoundReport:
    if not words:
        return RoundReport(passes, 0, stats.dropped, stats.looped,
                           0.0, 0.0, 0.0, None if previous is None else 1.0)
    unanimous = sum(1 for w in words if w.votes == passes) / len(words)
    return RoundReport(
        passes=passes,
        words=len(words),
        dropped=stats.dropped,
        looped=stats.looped,
        mean_confidence=sum(w.confidence for w in words) / len(words),
        mean_agreement=sum(w.agreement for w in words) / len(words),
        unanimous=unanimous,
        stability=None if previous is None else text_stability(previous, words),
    )


def text_stability(before: list[ConsensusWord], after: list[ConsensusWord]) -> float:
    """1 − word error rate between two consensus results.

    This is the convergence signal: once an extra pass moves this above the
    configured threshold, more passes are buying nothing and the loop stops.
    """
    a = [normalize_token(w.text) for w in before]
    b = [normalize_token(w.text) for w in after]
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    # Levenshtein over word tokens, two rows at a time.
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, token_b in enumerate(b, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (0 if token_a == token_b else 1),
            )
        previous = current
    return max(0.0, 1.0 - previous[len(b)] / max(len(a), len(b)))


def to_segments(words: list[ConsensusWord], *,
                language: str | None = None,
                language_at: Callable[[float, float], str | None] | None = None,
                ) -> list[dict]:
    """Group consensus words into Whisper-shaped segments.

    The output matches the raw Whisper JSON schema on purpose: SRT/VTT export
    and the Lyrixa bridge then work on consensus output without a single
    special case.

    `language_at` labels each segment with the language actually sung over it.
    On a DJ set that is not one value, and the translator downstream needs to
    know which line came from which language to pick its route.
    """
    if not words:
        return []

    groups: list[list[ConsensusWord]] = [[words[0]]]
    for previous, word in zip(words, words[1:]):
        current = groups[-1]
        gap = word.start - previous.end
        span_with_word = word.end - current[0].start
        clause_span = previous.end - current[0].start
        punctuation_break = bool(_STRONG_END.search(previous.text))
        # A comma ends a line once there is a useful clause on its left.  The
        # small minimum avoids turning lyrics such as "Then, then, then," into
        # one subtitle per word.
        punctuation_break = punctuation_break or (
            bool(_CLAUSE_END.search(previous.text))
            and (len(current) >= SEGMENT_MIN_CLAUSE_WORDS
                 or clause_span >= SEGMENT_MIN_CLAUSE_SECONDS)
        )
        too_long = (
            span_with_word > SEGMENT_MAX_SECONDS
            or len(current) >= SEGMENT_MAX_WORDS
            or len(join_words([w.text for w in current] + [word.text]))
            > SEGMENT_MAX_CHARS
        )
        if gap >= SEGMENT_PAUSE or punctuation_break or too_long:
            groups.append([word])
        else:
            current.append(word)

    segments: list[dict] = []
    for group in groups:
        tokens = [w.text for w in group]
        if is_filler_line(join_words(tokens)):
            continue
        # Each token carries the space that precedes it, exactly like Whisper's
        # own `words[]`, so every consumer reconstructs the same line by
        # concatenation and none of them re-guesses the spacing.
        prefixes = space_prefixes(tokens)
        segments.append({
            "id": len(segments),
            "start": round(group[0].start, 3),
            "end": round(group[-1].end, 3),
            "text": join_words(tokens),
            "words": [
                {
                    "word": f"{prefix}{w.text}",
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "probability": round(w.confidence, 4),
                    "agreement": round(w.agreement, 4),
                    "votes": w.votes,
                }
                for prefix, w in zip(prefixes, group)
            ],
            "confidence": round(
                sum(w.confidence for w in group) / len(group), 4),
            "agreement": round(
                sum(w.agreement for w in group) / len(group), 4),
        })
        spoken = (language_at(group[0].start, group[-1].end)
                  if language_at else language)
        if spoken:
            segments[-1]["language"] = spoken
    return segments
