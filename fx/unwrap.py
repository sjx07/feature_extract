"""Harvest residue: a prompt that arrived wrapped in the code or record it was lifted from.

    unwrap('parser.add_argument("--system_prompt", default="You are a SQLite expert...")')
        -> ("You are a SQLite expert...", "python string literal")

A harvester sometimes keeps the line of code, the JSON record or the escaped one-line string around
the prompt instead of the prompt the model receives. Decomposing that would type the wrapper as
material and, worse, locate atoms in the escaped form (literal \\n, \\") so the same prompt harvested
cleanly elsewhere would never match. `unwrap` takes the prompt out when the shape is one it can
resolve exactly, and says how; the importer keeps the original beside it. Anything else is left as
it is. No model call.

Shapes: a JSON record whose one big string field is the prompt; Python source (an assignment, an
argparse default, a docstring, an f-string with {slots}, adjacent-string concatenation) whose
largest string literal is the prompt; a Lean s!"..." literal; and a one-line string with escaped
newlines and no wrapper at all. After any of those, or alone: code fences that a paper's typesetting
mangled from ``` to "` (on a line of their own, or inside \verb|"`|) are put back, so the model can
quote them and the fence reads as a fence.
"""
from __future__ import annotations

import ast
import json
import re
import textwrap
from typing import Optional

# the first non-empty line of something that is code, not prose
CODE_HEAD = re.compile(r"^\s*(parser\.add_argument|def |class |import |from \w+ import|@\w+|[A-Za-z_][\w.\[\]]*\s*(=|:=|\+=)\s*|let \w+|const |let |var |return |\{\s*\"|\{$|\[$)")
ESCAPES = re.compile(r"\\n|\\t|\\\"|\\'|\\\\")
_UNESC = {"\\n": "\n", "\\t": "\t", '\\"': '"', "\\'": "'", "\\\\": "\\"}
MIN_SHARE = 0.4          # the candidate must be most of the text, or it is a literal inside a real prompt


def _big_enough(candidate: str, text: str) -> bool:
    return len(candidate.strip()) >= 40 and len(candidate) >= MIN_SHARE * len(text)


def _finish(s: str) -> str:
    return textwrap.dedent(s).strip("\n")


def _from_json(text: str) -> Optional[tuple[str, str]]:
    try:
        obj = json.loads(text)
    except ValueError:
        return None
    if isinstance(obj, str):
        return obj, "json string"
    if isinstance(obj, dict):
        strings = [(k, v) for k, v in obj.items() if isinstance(v, str)]
        if strings:
            k, v = max(strings, key=lambda kv: len(kv[1]))
            if _big_enough(v, text):
                return v, f"json field {k}"
    return None


def _joined(node: ast.JoinedStr) -> str:
    out = []
    for v in node.values:
        if isinstance(v, ast.Constant):
            out.append(str(v.value))
        elif isinstance(v, ast.FormattedValue):
            out.append("{" + ast.unparse(v.value) + "}")
    return "".join(out)


def _from_python(text: str) -> Optional[tuple[str, str]]:
    try:
        tree = ast.parse(textwrap.dedent(text))
    except SyntaxError:
        return None
    best: Optional[tuple[str, str]] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            cand, how = node.value, "python string literal"
        elif isinstance(node, ast.JoinedStr):
            cand, how = _joined(node), "python f-string"
        else:
            continue
        if best is None or len(cand) > len(best[0]):
            best = (cand, how)
    if best and _big_enough(best[0], text):
        return best
    return None


_TRIPLE = re.compile(r'("""|\'\'\')(.*?)\1', re.S)
_LEAN = re.compile(r's!"((?:[^"\\]|\\.)*)"', re.S)
_DQ = re.compile(r'"((?:[^"\\\n]|\\.)*)"')


def _from_fragment(text: str) -> Optional[tuple[str, str]]:
    """Code that is not a complete Python module: the largest quoted literal by three regexes, escapes resolved."""
    cands: list[tuple[str, str]] = []
    for m in _TRIPLE.finditer(text):
        cands.append((m.group(2), "triple-quoted literal"))
    for m in _LEAN.finditer(text):
        cands.append((_unescape(m.group(1)), "lean s! literal"))
    for m in _DQ.finditer(text):
        cands.append((_unescape(m.group(1)), "quoted literal"))
    if not cands:
        return None
    cand, how = max(cands, key=lambda c: len(c[0]))
    return (cand, how) if _big_enough(cand, text) else None


def _unescape(s: str) -> str:
    return ESCAPES.sub(lambda m: _UNESC[m.group(0)], s)


_FENCE_LINE = re.compile(r'(?m)^([ \t]*)"`([A-Za-z0-9_+#.-]*)[ \t]*$')
_FENCE_VERB = re.compile(r'\\verb\|"`\|')


def mend_fences(text: str) -> tuple[str, bool]:
    """``` mangled to "` by typesetting: only a "` alone on a line (with an optional language word) or inside
    \\verb|"`| is a fence; an inline "`name`" is left alone."""
    if "```" in text:
        return text, False
    out = _FENCE_VERB.sub("```", text)
    out = _FENCE_LINE.sub(lambda m: m.group(1) + "```" + m.group(2), out)
    return out, out != text


def unwrap(text: str) -> tuple[str, Optional[str]]:
    """(the prompt as the model receives it, how it was unwrapped) or (text, None) when nothing applied."""
    stripped = text.strip()
    if not stripped:
        return text, None
    r = _from_json(stripped)
    if r is None and CODE_HEAD.match(stripped):
        r = _from_python(stripped) or _from_fragment(stripped)
    if r is None and "\n" not in stripped and len(ESCAPES.findall(stripped)) >= 3:
        r = (_unescape(stripped), "escaped one-line string")
    out, how = (text, None) if r is None else (_finish(r[0]), r[1])
    if not out.strip() or out == stripped:
        out, how = text, None
    out, mended = mend_fences(out)
    if mended:
        how = f"{how} + mangled fences" if how else "mangled fences"
    return out, how
