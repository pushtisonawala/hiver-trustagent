from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict

    # convenience accessors -------------------------------------------------
    @property
    def brand(self) -> str:
        return self.raw["brand"]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, key: str) -> Path:
        return ROOT / self.raw["paths"][key]

    @property
    def intent_labels(self) -> list[str]:
        return [d["name"] for d in self.raw["intents"]["labels"]]

    @property
    def intent_risk(self) -> dict[str, str]:
        return {d["name"]: d["risk"] for d in self.raw["intents"]["labels"]}

    def model(self, role: str) -> dict:
        return self.raw["models"][role]

    def __getitem__(self, key):
        return self.raw[key]


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> Config:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p) as fh:
        return Config(raw=yaml.safe_load(fh))


def offline() -> bool:
    """True if we must not call any paid API."""
    if os.environ.get("TRUSTAGENT_OFFLINE", "0") == "1":
        return True
    keys = ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    return not any(os.environ.get(k) for k in keys)
