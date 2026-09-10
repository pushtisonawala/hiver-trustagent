"""Draft a reply grounded in retrieved, historically-successful brand replies."""
from __future__ import annotations

import json

from . import llm
from .config import Config

VOICE = (
    "You are a Spotify customer-support agent replying on Twitter/X. "
    "Voice: warm, concise, lowercase-friendly, no corporate jargon, 1-3 sentences, "
    "at most one emoji, never promise refunds or account actions you cannot verify. "
    "Only state fixes that appear in the EVIDENCE. If the issue needs account-specific "
    "data (email, order id, payment details), do NOT ask for it in public — say you'll "
    "continue in DM. If the evidence does not cover the problem, set grounded=false."
)


def _clip(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + "…"


def build_prompt(customer_text: str, intent: str, hits: list[dict]) -> str:
    ev = []
    for i, h in enumerate(hits, 1):
        ev.append(
            f"[{i}] sim {h['similarity']:.2f} / resolved {h['resolution_score']:.2f}\n"
            f"    they said: {_clip(h['past_customer'], 200)}\n"
            f"    brand:     {_clip(h['past_reply'], 240)}"
        )
    return (
        f"INTENT: {intent}\n\nCUSTOMER MESSAGE:\n{_clip(customer_text, 500)}\n\n"
        f"EVIDENCE (past resolved cases):\n" + "\n".join(ev) +
        "\n\nReturn JSON: {\"reply\": str, \"used_evidence\": [int], "
        "\"grounded\": bool, \"missing_info\": [str]}"
    )


def draft_reply(customer_text: str, intent: str, hits: list[dict], cfg: Config) -> dict:
    m = cfg.model("agent")
    try:
        raw = llm.complete(
            m["provider"], m["model"], VOICE,
            build_prompt(customer_text, intent, hits),
            max_tokens=900, json_mode=True, reasoning="none",
        )
        d = llm.parse_json(raw)
        d.setdefault("reply", "")
        d.setdefault("used_evidence", [])
        d.setdefault("grounded", bool(hits))
        d.setdefault("missing_info", [])
        # map evidence indices back to thread ids
        d["used_evidence"] = [
            hits[i - 1]["thread_id"] for i in d["used_evidence"]
            if isinstance(i, int) and 1 <= i <= len(hits)
        ]
        return d
    except Exception:  # noqa: BLE001 — offline / rate-limit / bad JSON -> heuristic draft
        from .heuristics import heuristic_draft

        return heuristic_draft(customer_text, hits)
