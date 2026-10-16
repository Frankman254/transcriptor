"""How word tokens become a line of text.

One rule, one place.  A Japanese song with an English hook is the normal case,
not the exception, and joining it needs a decision *per boundary*: 「終わらない」
runs into the next kanji with no space, but "don't let me grow" falls apart
without them.  Deciding once per track — the way a single `cjk` flag does —
produces `Tellmeso,but,but,don'tletmegrow`.
"""

from __future__ import annotations

# Interfaz externa de este módulo; lo que no está aquí es seam interno
# y solo lo tocan sus propias pruebas (docs/arquitectura.md).
__all__ = [
    "display_width", "is_no_space_char", "join_raw", "join_tokens",
    "space_prefixes",
]

import re

# Scripts written without spaces between words.
_NO_SPACE_SCRIPT = re.compile(
    "[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]"
)
# Full-width punctuation belongs to the same no-space world.
_CJK_PUNCTUATION = "。、「」『』（）〜・！？：；"
# Latin punctuation that hugs the word before it.
_TRAILING_PUNCTUATION = ",.;:!?…)]}»'\"’”"
# ...and the word after it.
_LEADING_PUNCTUATION = "([{«¿¡"

_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?…)\]}])")


def is_no_space_char(char: str) -> bool:
    return bool(_NO_SPACE_SCRIPT.match(char)) or char in _CJK_PUNCTUATION


def needs_space(previous: str, current: str) -> bool:
    """Should a space go between these two adjacent tokens?"""
    if not previous or not current:
        return False
    left, right = previous[-1], current[0]
    # Two space-free characters run together — but only both. A kanji that
    # follows a Latin word starts a new word, so `go` + `誰の` needs its space.
    if is_no_space_char(left) and is_no_space_char(right):
        return False
    # A token that is only punctuation still attaches to its neighbour, in
    # either script: `word` + `,` and `語` + `。` both close up.
    if right in _TRAILING_PUNCTUATION or right in _CJK_PUNCTUATION:
        return False
    if left in _LEADING_PUNCTUATION:
        return False
    return True


def space_prefixes(tokens: list[str]) -> list[str]:
    """The leading space (or "") each token needs to reconstruct the line.

    Returned separately so callers can store Whisper-style tokens — where the
    space is part of the token — without re-deriving the rule.
    """
    prefixes = [""]
    for previous, current in zip(tokens, tokens[1:]):
        prefixes.append(" " if needs_space(previous, current) else "")
    return prefixes


def join_tokens(tokens: list[str]) -> str:
    """Join stripped tokens into one line."""
    clean = [token.strip() for token in tokens if token and token.strip()]
    if not clean:
        return ""
    prefixes = space_prefixes(clean)
    joined = "".join(prefix + token for prefix, token in zip(prefixes, clean))
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", joined).strip()


def display_width(text: str) -> int:
    """Roughly how much screen a line takes, in Latin-character units.

    A kanji occupies two columns and carries about as much meaning as a short
    English word, so measuring a Japanese line in code points makes it look
    twice as short as it reads.  Line-length budgets are about what fits on
    screen, so they have to be measured in width, not in characters.
    """
    return sum(2 if is_no_space_char(char) else 1 for char in text)


def join_raw(tokens: list[str]) -> str:
    """Join tokens that may already carry their own leading whitespace.

    Whisper emits `["Esta", " noche"]`; hand-written fixtures and other
    frontends emit `["Esta", "noche"]`.  Both must produce the same line.
    """
    raw = [token for token in tokens if token and token.strip()]
    if not raw:
        return ""
    if any(token[:1].isspace() for token in raw[1:]):
        return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", "".join(raw)).strip()
    return join_tokens(raw)
