"""Multi-provider LLM wrapper with an on-disk cache and free-tier throttling.

Providers (all reachable through the OpenAI client with a different base_url):
  * groq    -> https://api.groq.com/openai/v1        FREE, fast (Llama models)
  * gemini  -> generativelanguage.../v1beta/openai/  FREE (Google Gemini)
  * openai  -> api.openai.com                        paid (optional)
  * anthropic                                        paid (optional)

Default config uses groq + gemini, so a full run costs $0. If no key is set (or
TRUSTAGENT_OFFLINE=1) callers fall back to heuristics (heuristics.py).

Every call is cached by a hash of its inputs, so re-runs are free and instant —
that is what makes the "<15 min to reproduce" promise real. Free tiers are
rate-limited (~30 req/min); we throttle per-provider and back off on 429 so the
*first* run just runs slower, it does not fail.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

CACHE_DIR = Path(os.environ.get("TRUSTAGENT_CACHE", ".cache/llm"))

# minimum seconds between calls to the same provider:model (free-tier friendly).
# Groq limits are per-model, so we throttle per-model; a bit of 429 + backoff on
# top of this is expected and handled.
# Groq free tier is ~30 req/min per model; 2.0s keeps us just under it so we
# don't burn time in 429 backoff. Each model is its own bucket, so they run
# concurrently. Override with TRUSTAGENT_GROQ_INTERVAL if your limits differ.
_MIN_INTERVAL = {
    "groq": float(os.environ.get("TRUSTAGENT_GROQ_INTERVAL", "2.05")),
    "gemini": 7.0, "openai": 0.2, "anthropic": 0.2,
}
_last_call: dict[str, float] = {}
_locks: dict[str, threading.Lock] = {}
_reg_lock = threading.Lock()


class OfflineError(RuntimeError):
    pass


def _lock_for(p: str) -> threading.Lock:
    with _reg_lock:
        return _locks.setdefault(p, threading.Lock())


def _throttle(provider: str, model: str):
    """Reserve a time-slot per provider:model, then sleep to it *outside* the
    lock so N workers stagger instead of serialising."""
    bucket = f"{provider}:{model}"
    with _lock_for(bucket):
        nxt = max(time.monotonic(), _last_call.get(bucket, 0.0) + _MIN_INTERVAL.get(provider, 0.2))
        _last_call[bucket] = nxt
    delay = nxt - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def _cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _cache_read(key: str):
    fp = CACHE_DIR / key[:2] / f"{key}.json"
    return json.loads(fp.read_text())["response"] if fp.exists() else None


def _cache_write(key: str, payload: dict, response: str):
    fp = CACHE_DIR / key[:2] / f"{key}.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps({"payload": payload, "response": response}, ensure_ascii=False))


_BASE_URL = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openai": None,
}
_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _get_key(provider: str) -> str | None:
    env = _KEY_ENV[provider]
    for name in (env if isinstance(env, tuple) else (env,)):
        if os.environ.get(name):
            return os.environ[name]
    return None


def complete(provider: str, model: str, system: str, user: str, *,
             max_tokens: int = 1024, json_mode: bool = False,
             temperature: float = 0.0, reasoning: str = "low", retries: int = 6) -> str:
    # NB: `reasoning` is deliberately NOT in the cache key — it changes latency/cost,
    # not the semantic request, and we don't want to invalidate the committed cache.
    payload = {"provider": provider, "model": model, "system": system, "user": user,
               "max_tokens": max_tokens, "json_mode": json_mode, "temperature": temperature}
    key = _cache_key(payload)
    cached = _cache_read(key)
    if cached is not None:
        return cached

    if os.environ.get("TRUSTAGENT_OFFLINE", "0") == "1":
        raise OfflineError("TRUSTAGENT_OFFLINE=1")
    if _get_key(provider) is None and provider != "anthropic":
        raise OfflineError(f"no key for {provider}")

    last_err = None
    for attempt in range(retries):
        try:
            _throttle(provider, model)
            if provider == "anthropic":
                text = _anthropic(model, system, user, max_tokens, temperature, json_mode)
            else:
                text = _openai_compatible(provider, model, system, user, max_tokens,
                                          temperature, json_mode, reasoning)
            _cache_write(key, payload, text)
            return text
        except OfflineError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e).lower()
            sleep = min(60, 3 * (2 ** attempt)) if ("429" in msg or "rate" in msg or "quota" in msg) else 2 ** attempt
            time.sleep(sleep)
    raise RuntimeError(f"LLM call failed after {retries} tries ({provider}/{model}): {last_err}")


def _openai_compatible(provider, model, system, user, max_tokens, temperature, json_mode,
                       reasoning="low"):
    from openai import OpenAI

    client = OpenAI(api_key=_get_key(provider), base_url=_BASE_URL.get(provider))
    # gpt-oss / qwen3 on Groq are reasoning models. Reasoning tokens count against
    # the 8k tokens/min free-tier cap, so callers pass reasoning="none" for cheap
    # tasks (classify, judge) and "low" only where it helps (drafting).
    is_reasoner = "gpt-oss" in model or "qwen3" in model
    mt = max(max_tokens, 1400) if (is_reasoner and reasoning != "none") else max_tokens
    kwargs = dict(
        model=model, max_tokens=mt, temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    if is_reasoner:
        kwargs["reasoning_effort"] = reasoning
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    def _call(kw):
        return client.chat.completions.create(**kw).choices[0].message.content or ""

    try:
        out = _call(kwargs)
    except Exception as e:  # noqa: BLE001
        m = str(e).lower()
        if "429" in m or "rate" in m or "quota" in m:
            raise  # let the outer loop back off
        # otherwise assume the endpoint rejected response_format / reasoning_effort
        kwargs.pop("response_format", None)
        kwargs.pop("reasoning_effort", None)
        kwargs["messages"][0]["content"] += "\n\nReturn ONLY a valid JSON object."
        out = _call(kwargs)
    if not out.strip():  # reasoning ate the whole budget — retry bigger
        kwargs["max_tokens"] = mt * 2
        kwargs.pop("reasoning_effort", None)
        out = _call(kwargs)
    return out


def _anthropic(model, system, user, max_tokens, temperature, json_mode):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise OfflineError("no ANTHROPIC_API_KEY")
    import anthropic

    client = anthropic.Anthropic()
    sys = system + ("\n\nRespond with a single valid JSON object and nothing else." if json_mode else "")
    kwargs = dict(model=model, max_tokens=max_tokens, system=sys,
                  messages=[{"role": "user", "content": user}])
    if "haiku" in model or "claude-3" in model:
        kwargs["temperature"] = temperature
    else:
        kwargs["thinking"] = {"type": "disabled"}
    try:
        msg = client.messages.create(**kwargs)
    except Exception:
        kwargs.pop("thinking", None)
        msg = client.messages.create(**kwargs)
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1:
            return json.loads(text[s : e + 1])
        raise
