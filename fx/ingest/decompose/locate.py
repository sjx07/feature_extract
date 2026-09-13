"""Exact quote resolution and structural validation of one span's components. A component is
located by its first and last words; the span is derived by exact containment under a monotonic
cursor after whitespace and punctuation normalisation. Nothing is fuzzy: every failure is named,
and a component that cannot be located keeps span=None. Ported from FACET."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from fx.ingest.decompose.contract import Component, Span
from fx.ingest.decompose.prompts import SHIELD

_WS = re.compile(r"\s+")
_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def normalize(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space and fold typographic quotes and dashes. Returns
    (normalized, index_map): index_map[i] is the offset in the original text of normalized char i."""
    out: list[str] = []
    idx: list[int] = []
    pending_space = False
    for i, ch in enumerate(text):
        ch = ch.translate(_QUOTE_MAP)
        if ch.isspace():
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            idx.append(i - 1)
            pending_space = False
        for piece in unicodedata.normalize("NFKC", ch):
            out.append(piece)
            idx.append(i)
    return "".join(out), idx


def _norm_quote(q: str) -> str:
    for a, b in SHIELD.items():                     # the model quotes the shielded tag it was shown
        q = q.replace(b, a)
    return _WS.sub(" ", unicodedata.normalize("NFKC", q.translate(_QUOTE_MAP))).strip()


def resolve(text: str, start: str, end: str, cursor: int = 0, limit: Optional[int] = None, *, norm=None) -> Optional[Span]:
    ntext, idx = norm or normalize(text)
    s, e = _norm_quote(start), _norm_quote(end)
    if not s or not e:
        return None
    ncur = 0
    while ncur < len(idx) and idx[ncur] < cursor:
        ncur += 1
    si = ntext.find(s, ncur)
    ei = ntext.find(e, si) if si >= 0 else -1
    if si < 0 or ei < 0:                      # a quote the model re-cased ("I Will ask"): match without case
        lt, ls, le = ntext.lower(), s.lower(), e.lower()
        si = lt.find(ls, ncur)
        ei = lt.find(le, si) if si >= 0 else -1
        if si < 0 or ei < 0:
            return None
    lo = idx[si]
    hi = idx[ei + len(e) - 1] + 1
    if limit is not None and hi > limit:
        return None
    return (lo, hi)


def occurrences(text: str, quote: str, lo: int, hi: int, *, norm=None) -> int:
    ntext, idx = norm or normalize(text)
    q = _norm_quote(quote)
    if not q:
        return 0
    a = 0
    while a < len(idx) and idx[a] < lo:
        a += 1
    b = a
    while b < len(idx) and idx[b] < hi:
        b += 1
    return ntext[a:b].count(q)


def validate_components(components: list[Component], text: str, lo: int, hi: int, path: str = "", *, norm=None) -> list[str]:
    """Locate one span's components inside [lo, hi) and return the named failures. An atom
    without facets is a failure; a section or material with facets keeps none of them."""
    failures: list[str] = []
    norm = norm or normalize(text)
    cursor = lo
    for i, part in enumerate(components):
        label = f"component[{path}{i}]"
        part.span = resolve(text, part.start, part.end, cursor, hi, norm=norm)
        if part.span is None:
            anywhere = resolve(text, part.start, part.end, 0, norm=norm)
            why = ("outside the span" if anywhere and (anywhere[0] < lo or anywhere[1] > hi) else "out of order" if anywhere else "not found")
            failures.append(f"{label}: {why}: {part.start!r} .. {part.end!r}")
            part.flags.append("unlocated")
            continue
        cursor = part.span[1]
        if part.kind == "atom" and not part.facets:
            failures.append(f"{label}: an atom but has no facets")
            part.flags.append("no_facets")
        if part.kind == "material" and not part.facets:
            failures.append(f"{label}: material but has no facet saying what it provides")
            part.flags.append("no_facets")
        if part.kind == "section" and part.facets:
            part.facets = []
    # A quote is resolved at its first occurrence after the previous component, so a repeated start is harmless
    # (components are consecutive). A repeated END is the real ambiguity: if the end quote occurs again before
    # the next located component begins, the component may extend further than the first occurrence.
    located = [(i, p) for i, p in enumerate(components) if p.span is not None]
    for k, (i, part) in enumerate(located):
        nxt = located[k + 1][1].span[0] if k + 1 < len(located) else hi
        if occurrences(text, part.end, part.span[1], nxt, norm=norm) > 0:
            failures.append(f"{path_label(path, i)}: the end quote {part.end!r} occurs again before the next component, so the component may reach further; quote its last five or six words exactly as written")
            part.flags.append("ambiguous_end")
    return failures


def path_label(path: str, i: int) -> str:
    return f"component[{path}{i}]"


def nonspace(text: str, lo: int, hi: int) -> int:
    return sum(1 for ch in text[lo:hi] if not ch.isspace())
