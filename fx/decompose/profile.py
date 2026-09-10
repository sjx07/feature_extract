"""profile(text, ask): recursive refinement of one prompt into atoms and material.

refine(span): one REFINE call -> components; validate; on failure one re-ask with the failures
named; then every section is refined in its own call. The recursion ends by definition (an atom
or material is a leaf), with one guard: a span that comes back as a single section covering
itself makes no progress, so it is asked once more for a single atom and otherwise kept as an
unrefined leaf. Leaves that look compound (two or more sentences) are refined once more.

Gaps: every unowned stretch that reads as a sentence or more is refined once as a span of its
own, whatever words it contains and wherever it sits. Each gap is recorded with its outcome.
The model alone decides what is an atom and what is material. What it leaves unowned or declines
is recorded as a gap, and the queues show it; nothing here second-guesses it.

`ask` is any callable prompt -> reply text; the stage wires it to fx.llm.Client.
"""
from __future__ import annotations

import re
import time
from typing import Callable, Optional

from . import prompts as P
from .contract import Component, Tree, parse_components
from .locate import nonspace, normalize, validate_components

NO_PROGRESS = 0.95
_SENT = re.compile(r"(?<=[.!?;])(?<!\d\.)(?<!\b[a-zA-Z]\.)\s+(?=[A-Z(\-*•\d\"“])")

Ask = Callable[[str], str]
_WORD = re.compile(r"[A-Za-z]{2,}")


def gap_worth_refining(text: str, lo: int, hi: int) -> bool:
    """A sentence or more of prose: at least four words with letters, and not a bare slot like {question}."""
    seg = text[lo:hi]
    if re.fullmatch(r"\s*[{<\[][^\n]{0,80}[}>\]]\s*", seg):
        return False
    return len(_WORD.findall(seg)) >= 4


def looks_compound(atom_text: str) -> bool:
    return len([x for x in _SENT.split(atom_text.strip()) if x.strip()]) >= 2


def _trim(text: str, lo: int, hi: int) -> tuple[int, int]:
    while lo < hi and text[lo].isspace():
        lo += 1
    while hi > lo and text[hi - 1].isspace():
        hi -= 1
    return lo, hi


def _gaps(text: str, components: list[Component], lo: int, hi: int):
    pos = lo
    for c in components:
        if c.span is None:
            continue
        if c.span[0] > pos:
            a, b = _trim(text, pos, c.span[0])
            if b > a:
                yield a, b
        pos = max(pos, c.span[1])
    if hi > pos:
        a, b = _trim(text, pos, hi)
        if b > a:
            yield a, b


def _no_progress(components, lo, hi) -> bool:
    return (len(components) == 1 and components[0].kind == "section" and components[0].span is not None
            and (components[0].span[1] - components[0].span[0]) >= NO_PROGRESS * (hi - lo))


def _check(components, text, lo, hi, path, norm) -> list[str]:
    if components is None:
        return ["reply was not a JSON object with a 'components' list"]
    failures = validate_components(components, text, lo, hi, path, norm=norm)
    if not components:
        failures.append("no components returned")
    elif _no_progress(components, lo, hi):
        failures.append("the single component covers the whole span: divide the span into its components, or mark it an atom and give its facets")
    return failures


class _Session:
    def __init__(self, tree: Tree, ask: Ask):
        self.tree, self.ask = tree, ask

    def call(self, prompt: str) -> str:
        t0 = time.time()
        reply = self.ask(prompt) or ""
        self.tree.calls += 1
        self.tree.seconds += time.time() - t0
        self.tree.replies.append(reply)
        return reply


def _better(c2, f2, c1, f1) -> bool:
    """Whether the re-ask's reply should replace the first: a parsed reply always beats an unparsed one
    (a refusal or broken JSON on the re-ask must never erase a usable first reply); then more located
    components; then fewer failures."""
    if c2 is None:
        return False
    if c1 is None:
        return True
    n2, n1 = sum(1 for c in c2 if c.span is not None), sum(1 for c in c1 if c.span is not None)
    return (n2, -len(f2)) > (n1, -len(f1))


def _ask(s: _Session, text, lo, hi, path, norm, parent=None):
    prompt = P.render_refine(text, lo, hi, parent)
    reply = s.call(prompt)
    components = parse_components(reply)
    failures = _check(components, text, lo, hi, path, norm)
    if failures and not reply.strip():                 # nothing came back (ceiling spent, or the prompt's call budget is gone): no re-ask
        s.tree.failures.append(f"{path or 'root'}: no reply")
        return [], failures
    if failures:
        s.tree.reasks.append({"path": path or "root", "failures": list(failures)})
        components2 = parse_components(s.call(P.with_reask(prompt, failures)))
        failures2 = _check(components2, text, lo, hi, path, norm)
        if _better(components2, failures2, components, failures):
            components, failures = components2, failures2
        s.tree.flags.append(f"reasked:{path or 'root'}")
    return components or [], failures


def _force_atom(s: _Session, text, lo, hi, path, norm, parent=None, kind: str = "atom"):
    """One call for the span as a single leaf of `kind` with its facets: an atom, or a material leaf with the facet saying what it provides."""
    what = "kind atom with its facets" if kind == "atom" else f"kind material ({kind} is its material kind) with the one facet saying what it provides"
    prompt = P.with_reask(P.render_refine(text, lo, hi, parent), [f"the span cannot be divided further: return exactly one component of {what}"])
    components = parse_components(s.call(prompt))
    if not components or len(components) != 1 or components[0].kind != ("atom" if kind == "atom" else "material") or not components[0].facets:
        return None
    if validate_components(components, text, lo, hi, path, norm=norm):
        return None
    components[0].flags.append("forced_atom")
    s.tree.flags.append(f"forced_atom:{path or 'root'}")
    return components


def _refine(s: _Session, text, lo, hi, path, norm, parent=None) -> list[Component]:
    components, fails = _ask(s, text, lo, hi, path, norm, parent)
    s.tree.failures.extend(fails)
    if _no_progress(components, lo, hi):
        forced = _force_atom(s, text, lo, hi, path, norm, parent)
        if forced is not None:
            return forced
        components[0].flags.append("unrefined")
        return components
    extra = []
    in_gap = path.endswith(".") and path.rstrip(".").split(".")[-1].startswith("g")
    for k, (glo, ghi) in enumerate(list(_gaps(text, components, lo, hi))):
        if in_gap:
            break                                   # no gap inside a gap
        if not gap_worth_refining(text, glo, ghi):
            s.tree.gaps.append({"lo": glo, "hi": ghi, "outcome": "skipped", "path": f"{path}g{k}"})
            continue
        found = _refine(s, text, glo, ghi, f"{path}g{k}.", norm, parent=(lo, hi))
        found = [c for c in found if c.span is not None and "unrefined" not in c.flags]
        kinds = {c.kind for c in found}
        outcome = "recovered" if "atom" in kinds or "section" in kinds else ("material" if kinds == {"material"} else "declined")
        s.tree.gaps.append({"lo": glo, "hi": ghi, "outcome": outcome, "path": f"{path}g{k}"})
        if found:
            extra.extend(found)
            s.tree.flags.append(f"gap_refined:{path or 'root'}{k}")
    if extra:
        components = sorted(components + extra, key=lambda c: c.span[0] if c.span else -1)
    for i, part in enumerate(components):
        if part.span is None or part.kind != "section" or part.children:
            continue
        part.children = _refine(s, text, part.span[0], part.span[1], f"{path}{i}.", norm, parent=(lo, hi))
    return components


def _facet_check(s: _Session, text, norm) -> None:
    """An atom or a material leaf the model emitted without facets: nothing below would ever ask for them, so one
    call on the leaf alone asks for the leaf with its facets (with reasoning off the root reply often skips them)."""
    for part, _, path in list(s.tree.walk()):
        if not (part.is_leaf and part.kind in ("atom", "material") and part.span is not None) or part.facets:
            continue
        lo, hi = part.span
        forced = _force_atom(s, text, lo, hi, f"{path}.", norm, parent=(0, len(text)), kind="atom" if part.kind == "atom" else (part.material or "reference"))
        if forced is not None:
            part.facets = forced[0].facets
            part.flags = [f for f in part.flags if f != "no_facets"] + ["facets_asked"]
            s.tree.flags.append(f"facets_asked:{path}")


def _split_check(s: _Session, text, norm) -> None:
    for part, _, path in list(s.tree.walk()):
        if not (part.is_leaf and part.kind == "atom" and part.span is not None) or "forced_atom" in part.flags:
            continue
        lo, hi = part.span
        if not looks_compound(text[lo:hi]):
            continue
        components, _ = _ask(s, text, lo, hi, f"{path}.", norm, parent=(0, len(text)))
        located = [p for p in components if p.span is not None and p.kind == "atom" and p.facets]
        if len(located) >= 2 and len(located) == len(components):
            part.children, part.kind, part.facets = located, "section", []
            s.tree.flags.append(f"split:{path}")
        elif len(components) == 1 and located and (located[0].span[1] - located[0].span[0]) < 0.8 * (hi - lo):
            part.span, part.facets, part.start, part.end = located[0].span, located[0].facets, located[0].start, located[0].end
            s.tree.flags.append(f"trim:{path}")
        else:
            part.flags.append("compound_confirmed")


def profile(text: str, ask: Ask) -> Tree:
    tree = Tree()
    s = _Session(tree, ask)
    norm = normalize(text)
    tree.components = _refine(s, text, 0, len(text), "", norm)
    if not any(c.span is not None for c in tree.components):
        # nothing could be located at the root: keep the whole prompt as one unrefined leaf so it is visible in the queues
        whole = Component(start=" ".join(text.split()[:3]), end=" ".join(text.split()[-3:]), kind="section")
        whole.span = (0, len(text)); whole.flags += ["unrefined", "unparsed"]
        tree.components = [whole]
    _facet_check(s, text, norm)
    _split_check(s, text, norm)
    return tree


def metrics(tree: Tree, text: str) -> dict:
    """Coverage over instruction text: atom characters over (text minus what the model called material)."""
    total = nonspace(text, 0, len(text))
    material = sum(nonspace(text, *p.span) for p, _ in tree.material())
    instruction = max(total - material, 0)
    covered = min(sum(nonspace(text, *p.span) for p, _ in tree.atoms()), instruction)
    atoms = tree.atoms()
    gaps: dict = {}
    for g in tree.gaps:
        gaps[g["outcome"]] = gaps.get(g["outcome"], 0) + 1
    return {"chars": len(text), "instruction_chars": instruction, "covered_chars": covered,
            "coverage": round(covered / instruction, 4) if instruction else 1.0,
            "material_share": round(material / total, 4) if total else 0.0,
            "n_atoms": len(atoms), "n_material": len(tree.material()),
            "n_readings": sum(len(p.facets) for p, _ in atoms), "atoms_without_facets": sum(1 for p, _ in atoms if not p.facets),
            "calls": tree.calls, "seconds": round(tree.seconds, 1), "reasks": len(tree.reasks), "gaps": gaps,
            "unrefined": sum(1 for p, _, _ in tree.walk() if "unrefined" in p.flags),
            "compound_confirmed": sum(1 for p, _, _ in tree.walk() if "compound_confirmed" in p.flags)}
