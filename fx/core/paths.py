"""Where things live: one workspace directory per project, the same for the CLI and the GUI.

    runs/<workspace>/
      store.db            the store (SQLite; -wal and -shm beside it while open)
      logs/serve.log      the site's log, rotated
      logs/job-<id>.log   one log per run: every finished prompt, every error, the traceback if it died
      uploads/<corpus>/   files dropped into the site or imported from the terminal, kept verbatim
      exports/            anything written out for use elsewhere

The workspace is `--workspace` on the CLI or FX_WORKSPACE, default runs/dev. Nothing under runs/
is committed; data/corpora/ holds the checked-in test corpora.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT = "runs/dev"


class Workspace:
    def __init__(self, root: "str | Path" = DEFAULT):
        self.root = Path(root)
        self.logs = self.root / "logs"
        self.uploads = self.root / "uploads"
        self.exports = self.root / "exports"
        for d in (self.root, self.logs, self.uploads, self.exports):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def store_path(self) -> Path:
        return self.root / "store.db"

    def job_log(self, jid: int) -> Path:
        return self.logs / f"job-{jid}.log"

    def upload_dir(self, corpus: str) -> Path:
        d = self.uploads / "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in corpus)
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def from_env(cls, arg: Optional[str] = None) -> "Workspace":
        return cls(arg or os.environ.get("FX_WORKSPACE", DEFAULT))

    def __repr__(self) -> str:
        return f"Workspace({self.root})"
