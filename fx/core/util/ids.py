"""Ids the way prompts carry them: a letter and a number (R12, F3, G1), read back against the set the code knows.
Anything else, a made-up id, a bare word, an id outside the set, is dropped rather than guessed."""
from __future__ import annotations

import re
from typing import Iterable, Optional

_ID = re.compile(r"^[A-Za-z]?(\d+)$")


def parse_ids(xs, valid: Iterable[int]) -> list[int]:
    valid = set(valid)
    out = []
    for x in xs if isinstance(xs, list) else []:
        m = _ID.match(str(x).strip())
        if m and int(m.group(1)) in valid:
            out.append(int(m.group(1)))
    return out


def parse_id(x, valid: Iterable[int]) -> Optional[int]:
    got = parse_ids([x], valid)
    return got[0] if got else None
