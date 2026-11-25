from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


@dataclass(frozen=True, slots=True)
class AppSettings:
    root: Path
    database_path: Path
    config_dir: Path
    contact: str
    ai_enabled: bool
    ai_base_url: str
    ai_model: str
    ai_api_key: str

    def effective_contact(self, session_contact: str | None = None) -> str:
        """A session override, including an explicit blank, never changes settings."""
        return (self.contact if session_contact is None else session_contact).strip()

    @classmethod
    def load(cls, root: Path) -> AppSettings:
        project_root = root.resolve()
        contact = os.getenv("SHU_NOTICE_CONTACT", "").strip()
        database_override = os.getenv("SHU_NOTICE_DATABASE_PATH", "").strip()
        database_path = (
            Path(database_override).expanduser().resolve()
            if database_override
            else project_root / "data" / "shu_insight.sqlite3"
        )
        return cls(
            root=project_root,
            database_path=database_path,
            config_dir=project_root / "config",
            contact=contact,
            ai_enabled=_env_flag("SHU_NOTICE_AI_ENABLED"),
            ai_base_url=os.getenv("SHU_NOTICE_AI_BASE_URL", "http://127.0.0.1:11434/v1").strip(),
            ai_model=os.getenv("SHU_NOTICE_AI_MODEL", "").strip(),
            ai_api_key=os.getenv("SHU_NOTICE_AI_API_KEY", "").strip(),
        )
