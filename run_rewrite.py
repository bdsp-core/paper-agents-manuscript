#!/usr/bin/env python3
"""
run_rewrite.py — the "Clarity and Grace" REWRITE pipeline.

This is the rewrite-and-verify half of the review loop. run_review.py *grades* a
manuscript; this script *improves* it, paragraph by paragraph, then verifies it did
no harm:

    for each prose paragraph:
        clarity  -> cohesion -> concision        (Williams' Style)
        fidelity gate: did claims/hedges/numbers/citations survive? FLAG -> revert

It then optionally re-runs run_review.py on the result with an iteration label, so the
score delta lands in the existing dashboard automatically.

Usage
-----
# Dry run: show which paragraphs would be rewritten (no API calls)
python run_rewrite.py paper.md --dry-run

# Rewrite -> paper.revised.md  (+ a change log), with the fidelity gate on
python run_rewrite.py paper.md

# Rewrite, then score the result against the original in the dashboard
python run_rewrite.py paper.md --review --iteration-label rewrite-1

Floats are skipped (headings, tables, figures, code, and reference/funding/etc.
sections). The fidelity gate is the backstop: anything risky reverts to the original.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import anthropic

import agents as agents_module
from agents import fidelity_verify
from rewrite_agents import PIPELINE

SKIP_SECTION = re.compile(
    r"\b(authors?|affiliations?|corresponding author|manuscript metadata|references|"
    r"bibliography|funding|competing interests?|conflicts? of interest|"
    r"author contributions?|acknowledg|data and code availability|data availability|"
    r"code availability|keywords)\b",
    re.IGNORECASE,
)

# A bold "label:" line, e.g. "**Short title:**" — metadata, not narrative.
_LABEL_LINE = re.compile(r"^\*\*[^*\n]{1,40}:\*\*")


def _is_heading(block: str) -> bool:
    return block.lstrip().startswith("#")


def _heading_text(block: str) -> str:
    return block.lstrip("#").strip()


def _is_prose(block: str, in_skip_section: bool) -> bool:
    """Conservative: rewrite only normal narrative paragraphs."""
    s = block.strip()
    if not s or in_skip_section:
        return False
    if s.startswith(("#", "|", "!", ">", "```", "\\", "$$", "- ", "* ", "+ ", "%")):
        return False  # heading, table, image, quote, code, latex, latex-comment, bullet list
    if s.startswith("**Table") or s.startswith("**Figure") or _LABEL_LINE.match(s):
        return False  # float caption or a "**Label:**" metadata line
    # markdown/grid table rows or rule lines
    if any(re.match(r"^\s*[|+]", ln) or re.match(r"^\s*[-=+]{3,}\s*$", ln)
           for ln in s.splitlines()):
        return False
    # numbered reference / list entry, e.g. "1. Hazell L, ..."
    if re.match(r"^\[?\d+[\].]\s", s):
        return False
    if len(s.split()) < 8:          # captions stubs, metadata lines, labels
        return False
    return True


def segment(text: str) -> list[dict]:
    """Split into (block, separator) units, classified prose vs skip."""
    parts = re.split(r"(\n[ \t]*\n)", text)
    units: list[dict] = []
    in_skip = False
    for i, chunk in enumerate(parts):
        if i % 2 == 1:                       # separator
            units.append({"kind": "sep", "text": chunk})
            continue
        if _is_heading(chunk):
            in_skip = bool(SKIP_SECTION.search(_heading_text(chunk)))
            units.append({"kind": "skip", "text": chunk})
            continue
        units.append({
            "kind": "prose" if _is_prose(chunk, in_skip) else "skip",
            "text": chunk,
        })
    return units


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Clarity-and-Grace rewrite pipeline (clarity -> cohesion -> concision -> fidelity gate)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    ap.add_argument("paper", nargs="?", help="Path to manuscript (.md/.txt/.tex)")
    ap.add_argument("-o", "--output", help="Output path (default: <stem>.revised<ext>)")
    ap.add_argument("--no-fidelity", action="store_true",
                    help="Skip the per-paragraph fidelity gate (NOT recommended)")
    ap.add_argument("--review", action="store_true",
                    help="After rewriting, run run_review.py on the result with --revised "
                         "pointing at the original (records the score delta)")
    ap.add_argument("--iteration-label", default=None,
                    help="Label passed to run_review.py when --review is set")
    ap.add_argument("--model", default=None, help="Override Claude model")
    ap.add_argument("--max-paragraphs", type=int, default=None,
                    help="Cap on paragraphs rewritten (for quick tests)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show the rewrite plan without calling the API")
    args = ap.parse_args()

    if not args.paper:
        ap.print_help(); sys.exit(1)
    paper_path = Path(args.paper)
    if not paper_path.exists():
        print(f"Error: paper not found: {paper_path}"); sys.exit(1)
    text = paper_path.read_text(encoding="utf-8")

    cfg_path = Path("review_config.json")
    config = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    if args.model:
        agents_module.MODEL = args.model

    units = segment(text)
    prose_idx = [i for i, u in enumerate(units) if u["kind"] == "prose"]
    if args.max_paragraphs is not None:
        prose_idx = prose_idx[: args.max_paragraphs]

    print(f"\n📄 {paper_path}  ({len(text.split()):,} words)")
    print(f"   {len(prose_idx)} prose paragraphs to rewrite "
          f"({sum(1 for u in units if u['kind']=='skip')} float/skip blocks preserved)")
    print(f"   pipeline: {' -> '.join(a.id for a in PIPELINE)}"
          f"{'' if args.no_fidelity else ' -> fidelity gate'}")

    if args.dry_run:
        print("\n[dry-run] paragraphs that would be rewritten:")
        for n, i in enumerate(prose_idx, 1):
            preview = " ".join(units[i]["text"].split())[:90]
            print(f"   {n:3d}. {preview}…")
        print("\n[dry-run] No API calls made.\n"); sys.exit(0)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set."); sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    change_log: list[str] = []
    flagged: list[str] = []
    t0 = time.time()
    for n, i in enumerate(prose_idx, 1):
        original = units[i]["text"]
        revised = original
        for agent in PIPELINE:
            revised = agent.rewrite(client, revised, config)
        gate = "off"
        if not args.no_fidelity:
            ok, report = fidelity_verify(client, original, revised, config)
            if not ok:
                flagged.append(f"### Paragraph {n} — fidelity FLAG (reverted)\n\n"
                               f"{report}\n\n**Original kept:**\n\n> {original.strip()}\n")
                revised = original          # revert: never ship a meaning change
                gate = "FLAG→reverted"
            else:
                gate = "PASS"
        units[i]["text"] = revised
        status = "· unchanged" if revised.strip() == original.strip() else "· revised"
        print(f"   [{n}/{len(prose_idx)}] {status}  (fidelity: {gate})")
        if revised.strip() != original.strip():
            change_log.append(f"### Paragraph {n}\n\n**Before:**\n\n> {original.strip()}\n\n"
                              f"**After:**\n\n> {revised.strip()}\n")

    revised_text = "".join(u["text"] for u in units)
    out_path = Path(args.output) if args.output else \
        paper_path.with_suffix(f".revised{paper_path.suffix}")
    out_path.write_text(revised_text, encoding="utf-8")

    log_path = out_path.with_name(out_path.stem + ".changelog.md")
    log_body = ["# Rewrite change log", "",
                f"Source: `{paper_path}`  →  `{out_path}`",
                f"Paragraphs rewritten: {len(change_log)} / {len(prose_idx)}  ·  "
                f"fidelity flags reverted: {len(flagged)}", "", "---", ""]
    log_body += change_log
    if flagged:
        log_body += ["", "## Fidelity flags (reverted to original)", ""] + flagged
    log_path.write_text("\n".join(log_body), encoding="utf-8")

    print(f"\n  ⏱  {time.time()-t0:.1f}s   ✏️  {len(change_log)} revised   "
          f"🛡  {len(flagged)} reverted by fidelity gate")
    print(f"  Revised:    {out_path}")
    print(f"  Change log: {log_path}")

    if args.review:
        label = args.iteration_label or "rewrite"
        print(f"\n  ▶ Scoring the rewrite (run_review.py, label '{label}') …")
        subprocess.run([sys.executable, "run_review.py", str(out_path),
                        "--revised", str(paper_path), "--iteration-label", label],
                       check=False)


if __name__ == "__main__":
    main()
