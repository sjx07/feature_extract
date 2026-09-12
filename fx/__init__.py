"""feature_extract: every stage reads and writes one SQLite store (fx.store); the GUI is a view of it."""
__version__ = "0.0.1"

from . import settings as _settings  # noqa: E402

_settings.load()        # the key file under home into the environment, before any endpoint reads it
