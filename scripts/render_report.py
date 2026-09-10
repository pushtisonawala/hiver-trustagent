#!/usr/bin/env python
"""Fill REPORT.md with real numbers -> REPORT_GENERATED.md.

Placeholders:
  {{ models.agent.intent.accuracy | .2f }}      dotted path into results/metrics.json
  <!-- INCLUDE:results/RESULTS.md -->            splice a file in verbatim
Anything unresolved is left as `??` and listed at the end so nothing is faked silently.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = json.load(open(ROOT / "results/metrics.json"))
missing = []


def get(path: str):
    cur = M
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def repl(m):
    expr = m.group(1).strip()
    fmt = None
    if "|" in expr:
        expr, fmt = [x.strip() for x in expr.split("|", 1)]
    val = get(expr)
    if val is None:
        missing.append(expr)
        return "`??`"
    if isinstance(val, list):
        if len(val) == 3 and all(isinstance(x, (int, float)) for x in val):
            return f"{val[0]:.2f}  (95% CI {val[1]:.2f}–{val[2]:.2f})"
        return ", ".join(str(x) for x in val)
    if fmt:
        try:
            return format(val, fmt)
        except (ValueError, TypeError):
            pass
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def include(m):
    fp = ROOT / m.group(1)
    return fp.read_text() if fp.exists() else f"_(missing {m.group(1)})_"


src = (ROOT / "REPORT.md").read_text()
src = re.sub(r"<!--\s*INCLUDE:(.+?)\s*-->", include, src)
src = re.sub(r"\{\{(.+?)\}\}", repl, src)

banner = ""
gp = ROOT / "golden/golden_set.csv"
if gp.exists() and "PROVISIONAL" in gp.read_text():
    banner = ("> ⚠️ **These numbers are from the auto-labelled PROVISIONAL golden set.**\n"
              "> Run `python scripts/build_golden_set.py --review` to hand-label it, then\n"
              "> `make all` again. The pipeline and every metric are real; only the\n"
              "> gold labels are placeholder.\n\n")

dst = ROOT / "REPORT_GENERATED.md"
dst.write_text(banner + src)
print("wrote", dst)
if missing:
    print("\nUNRESOLVED placeholders (left as ??):")
    for x in sorted(set(missing)):
        print("  -", x)
