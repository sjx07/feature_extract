"""Corpora and prompts in the store, and the importers the GUI's drop zone and `fx import` share.

    import_path(store, "prompts.jsonl", name="text2sql", domain="text2sql")   FACET-format jsonl
    import_path(store, "folder_or_zip/", name="mine")                          one prompt per text file
    import_text(store, "You are ...", name="scratch")                          one pasted prompt

Prompts are keyed by their text hash inside a corpus, so a re-import of the same file adds
nothing, and the same text in two corpora is two prompt rows that share a hash (the GUI shows
the duplicate count). Source metadata is kept as recorded; the task label is unknown unless the
source carried one. A prompt that arrived inside the code or record it was harvested from is
unwrapped first (see fx.unwrap); the original is kept in meta["harvest"].
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Iterable, Optional

from .store import Store, now
from .unwrap import unwrap

TEXT_SUFFIXES = {".txt", ".md", ".prompt", ".jinja", ".j2", ".yaml", ".yml", ".json"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def corpus_id(store: Store, name: str) -> int:
    """The id of an existing corpus; KeyError when there is none (nothing is created)."""
    r = store.one("SELECT id FROM corpus WHERE name=?", (name,))
    if not r:
        raise KeyError(f"no corpus {name!r}")
    return int(r["id"])


def corpus_domain(store: Store, cid: int, default: str) -> str:
    r = store.one("SELECT domain FROM prompt WHERE corpus=? AND domain IS NOT NULL", (cid,))
    return (r["domain"] if r else None) or default


def get_corpus(store: Store, name: str, source: str = "") -> int:
    r = store.one("SELECT id FROM corpus WHERE name=?", (name,))
    if r:
        return int(r["id"])
    return store.insert("corpus", {"name": name, "source": source, "at": now()})


def record_import(store: Store, corpus_id: int, kind: str, path: Optional[str], domain: Optional[str] = None) -> int:
    """The import event, before its prompts land; `add_prompts` fills in the counts."""
    return store.insert("import", {"corpus": corpus_id, "kind": kind, "path": path, "domain": domain, "at": now()})


def imports(store: Store, corpus: Optional[str] = None, limit: int = 50) -> list[dict]:
    return [dict(r) for r in store.rows("SELECT i.*, k.name corpus_name FROM import i JOIN corpus k ON k.id=i.corpus" + (" WHERE k.name=?" if corpus else "") + " ORDER BY i.id DESC LIMIT ?",
                                        ([corpus] if corpus else []) + [limit])]


def add_prompts(store: Store, corpus_id: int, rows: Iterable[dict], imp: Optional[int] = None) -> dict:
    """rows: {text, id?, domain?, system?, task?, source_id?, meta?}. Returns counts, written to the import row too."""
    have = {r["sha"] for r in store.rows("SELECT sha FROM prompt WHERE corpus=?", (corpus_id,))}
    elsewhere = {r["sha"] for r in store.rows("SELECT sha FROM prompt WHERE corpus!=?", (corpus_id,))}
    added = skipped = dup = unwrapped = 0
    for r in rows:
        text = (r.get("text") or "").strip("\n")
        if not text.strip():
            skipped += 1
            continue
        meta = dict(r.get("meta") or {})
        text, how = unwrap(text)
        if how:
            meta["harvest"] = {"wrapped": how, "original": (r.get("text") or "").strip("\n")}
        h = sha(text)
        if h in have:
            skipped += 1
            continue
        pid = r.get("id") or f"{corpus_id}:{h[:16]}"
        if store.one("SELECT 1 FROM prompt WHERE id=?", (pid,)):
            pid = f"{corpus_id}:{h[:16]}"
        store.insert("prompt", {"id": pid, "corpus": corpus_id, "sha": h, "text": text, "domain": r.get("domain"), "system": r.get("system"),
                                "task": r.get("task"), "source_id": r.get("source_id"), "meta": meta, "at": now(), "import": imp})
        have.add(h)
        added += 1
        unwrapped += bool(how)
        if h in elsewhere:
            dup += 1
    if imp is not None:
        with store.lock:
            store.con.execute("UPDATE import SET added=?, skipped=?, unwrapped=? WHERE id=?", (added, skipped, unwrapped, imp)); store.con.commit()
    return {"added": added, "skipped": skipped, "duplicates_elsewhere": dup, "unwrapped": unwrapped, "import": imp}


def _facet_rows(path: Path, domain: Optional[str]) -> Iterable[dict]:
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if domain and r.get("domain") != domain:
            continue
        if r.get("is_prompt") is False:
            continue
        prov = r.get("provenance") or {}
        yield {"id": r.get("prompt_id") or r.get("id"), "text": r.get("text", ""), "domain": r.get("domain"),
               "system": r.get("system_id") or r.get("system"), "task": (r.get("labels") or {}).get("task") or r.get("task"),
               "source_id": r.get("record_id"),
               "meta": {k: r[k] for k in ("collection", "bank_source", "role", "stage", "subtask", "family", "use_case") if k in r} | {"provenance": prov}}


def _file_rows(files: Iterable[tuple[str, bytes]], domain: Optional[str] = None) -> Iterable[dict]:
    for name, data in files:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        yield {"id": None, "text": text, "source_id": name, "domain": domain, "meta": {"file": name}}


def import_path(store: Store, path: "str | Path", name: str, domain: Optional[str] = None) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    cid = get_corpus(store, name, str(p))
    if p.is_file() and p.suffix == ".jsonl":
        imp = record_import(store, cid, "jsonl", str(p), domain)
        return {"corpus": name, **add_prompts(store, cid, _facet_rows(p, domain), imp)}
    if p.is_file() and p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            files = [(n, z.read(n)) for n in z.namelist() if Path(n).suffix in TEXT_SUFFIXES and not n.endswith("/")]
        imp = record_import(store, cid, "zip", str(p), domain)
        return {"corpus": name, **add_prompts(store, cid, _file_rows(files, domain), imp)}
    if p.is_dir():
        files = [(str(f.relative_to(p)), f.read_bytes()) for f in sorted(p.rglob("*")) if f.is_file() and f.suffix in TEXT_SUFFIXES]
        imp = record_import(store, cid, "folder", str(p), domain)
        return {"corpus": name, **add_prompts(store, cid, _file_rows(files, domain), imp)}
    imp = record_import(store, cid, "file", str(p), domain)
    return {"corpus": name, **add_prompts(store, cid, _file_rows([(p.name, p.read_bytes())], domain), imp)}


def import_upload(store: Store, filename: str, data: bytes, name: str, domain: Optional[str] = None) -> dict:
    """The GUI drop zone: a jsonl, a zip, or one text file, as bytes."""
    cid = get_corpus(store, name, filename)
    suffix = Path(filename).suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for line in data.decode("utf-8").split("\n"):
            if line.strip():
                rows.append(json.loads(line))
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
            fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
            tmp = fh.name
        imp = record_import(store, cid, "jsonl", filename, domain)
        return {"corpus": name, **add_prompts(store, cid, _facet_rows(Path(tmp), domain), imp)}
    if suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            files = [(n, z.read(n)) for n in z.namelist() if Path(n).suffix in TEXT_SUFFIXES and not n.endswith("/")]
        imp = record_import(store, cid, "zip", filename, domain)
        return {"corpus": name, **add_prompts(store, cid, _file_rows(files, domain), imp)}
    imp = record_import(store, cid, "file", filename, domain)
    return {"corpus": name, **add_prompts(store, cid, _file_rows([(filename, data)], domain), imp)}


def import_text(store: Store, text: str, name: str = "scratch", domain: Optional[str] = None) -> dict:
    cid = get_corpus(store, name, "pasted")
    imp = record_import(store, cid, "paste", None, domain)
    return {"corpus": name, **add_prompts(store, cid, [{"id": None, "text": text, "domain": domain, "meta": {"pasted": True}}], imp)}


def corpora(store: Store) -> list[dict]:
    return [dict(r) for r in store.rows("SELECT c.*, (SELECT COUNT(*) FROM prompt p WHERE p.corpus=c.id) n_prompts, "
                                        "(SELECT SUM(LENGTH(text)) FROM prompt p WHERE p.corpus=c.id) chars FROM corpus c ORDER BY c.id DESC")]
