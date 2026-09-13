"""The tree a REFINE reply builds, and its parser. Components are atoms, sections, or material;
atoms carry facets. Tolerant parsing: vocabulary violations are coerced and flagged, never dropped."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from fx.core.util.jsonx import as_str, extract_object
from fx.ingest.decompose.prompts import MATERIAL_KINDS

Span = tuple[int, int]
POLARITIES = ("require", "forbid")
KINDS = ("atom", "section", "material")


@dataclass
class Facet:
    verb: str
    object: str = ""
    qualifier: str = ""
    polarity: str = "require"
    condition: str = "always"
    domain_terms: list[str] = field(default_factory=list)

    @property
    def declaration(self) -> str:
        return " ".join(x for x in (self.verb, self.object, self.qualifier) if x).strip()

    def to_dict(self) -> dict[str, Any]:
        return {"verb": self.verb, "object": self.object, "qualifier": self.qualifier, "polarity": self.polarity,
                "condition": self.condition, "domain_terms": list(self.domain_terms), "declaration": self.declaration}


@dataclass
class Component:
    start: str
    end: str
    kind: str = "section"                       # atom | section | material
    material: Optional[str] = None              # for kind material
    facets: list[Facet] = field(default_factory=list)
    span: Optional[Span] = None                 # filled by locate.validate_components
    children: list["Component"] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def atomic(self) -> bool:
        return self.kind == "atom"

    @property
    def readings(self) -> list[Facet]:
        """The facets recorded for this leaf, all in one shape: an atom's guidance; for material the one facet
        saying what the prompt provides; nothing for a section. Every facet is the model's."""
        if self.kind == "atom":
            return self.facets
        if self.kind == "material":
            return self.facets[:1]
        return []

    @property
    def is_leaf(self) -> bool:
        return self.kind != "section" or not self.children

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "kind": self.kind, "material": self.material,
                "span": list(self.span) if self.span else None, "flags": list(self.flags),
                "facets": [f.to_dict() for f in self.facets], "children": [c.to_dict() for c in self.children]}


@dataclass
class Tree:
    components: list[Component] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)      # prompt-level: reasked:<path>, split:<path>, gap_refined:<path>
    calls: int = 0
    seconds: float = 0.0
    replies: list[str] = field(default_factory=list)
    reasks: list[dict] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)       # {lo, hi, outcome, path}
    failures: list[str] = field(default_factory=list)

    def walk(self) -> Iterator[tuple[Component, int, str]]:
        stack = [(p, 1, str(i)) for i, p in reversed(list(enumerate(self.components)))]
        while stack:
            part, depth, path = stack.pop()
            yield part, depth, path
            for j, c in reversed(list(enumerate(part.children))):
                stack.append((c, depth + 1, f"{path}.{j}"))

    def leaves(self) -> list[tuple[Component, str]]:
        return [(p, path) for p, _, path in self.walk() if p.is_leaf and p.span is not None]

    def atoms(self) -> list[tuple[Component, str]]:
        return [(p, path) for p, path in self.leaves() if p.kind == "atom"]

    def material(self) -> list[tuple[Component, str]]:
        return [(p, path) for p, path in self.leaves() if p.kind == "material"]

    def to_dict(self) -> dict[str, Any]:
        return {"components": [p.to_dict() for p in self.components], "flags": list(self.flags), "calls": self.calls,
                "seconds": self.seconds, "reasks": list(self.reasks), "gaps": list(self.gaps), "failures": list(self.failures)}


# ---------------------------------------------------------------- parsing
def _facet(d: Any) -> Optional[Facet]:
    if not isinstance(d, dict):
        return None
    verb = as_str(d.get("verb")) or as_str(d.get("declaration"))
    if not verb:
        return None
    terms = d.get("domain_terms")
    return Facet(verb=verb, object=as_str(d.get("object")), qualifier=as_str(d.get("qualifier")),
                 polarity=as_str(d.get("polarity")).lower() or "require", condition=as_str(d.get("condition")) or "always",
                 domain_terms=[as_str(x) for x in terms] if isinstance(terms, list) else [])


def parse_components(reply: str) -> Optional[list[Component]]:
    """None when the reply is not a JSON object with a 'components' list."""
    obj = extract_object(reply, "components")
    if obj is None:
        return None
    out: list[Component] = []
    for d in obj["components"]:
        if not isinstance(d, dict):
            continue
        kind = as_str(d.get("kind")).lower()
        if kind not in KINDS:
            kind = "atom" if d.get("atomic") else ("material" if d.get("material") else "section")
        part = Component(start=as_str(d.get("start")), end=as_str(d.get("end")), kind=kind)
        if kind == "material":
            mk = as_str(d.get("material")).lower() or "other"
            if mk not in MATERIAL_KINDS:
                part.flags.append(f"bad_material:{mk}")
                mk = "other"
            part.material = mk
        raw = d.get("facets")
        for fd in (raw if isinstance(raw, list) else []):
            f = _facet(fd)
            if f is None:
                continue
            if f.polarity not in POLARITIES:
                part.flags.append(f"bad_polarity:{f.polarity}")
                f.polarity = "require"
            part.facets.append(f)
        out.append(part)
    return out
