#!/usr/bin/env python3
"""
run_review.py — CLI entry point for the paper-review-agents pipeline.

Usage examples
--------------
# Full review of a paper
python run_review.py paper.txt

# With code for reproducibility checking
python run_review.py paper.txt --code-file analysis.py

# Run only specific agents
python run_review.py paper.txt --agents vsnc,intro,paragraphs

# Skip Scholar-dependent agents (faster, no rate-limit risk)
python run_review.py paper.txt --no-scholar

# List all available agents
python run_review.py --list-agents

# Dry run (show what would run, don't call API)
python run_review.py paper.txt --dry-run

Reports are saved to reports/review_YYYYMMDD_HHMMSS.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

from agents import ALL_AGENTS, AGENT_REGISTRY, Context, AgentResult

# ---------------------------------------------------------------------------
# ANSI colour helpers (degrade gracefully on Windows)
# ---------------------------------------------------------------------------
_IS_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text

def bold(t):   return _c("1", t)
def dim(t):    return _c("2", t)
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def red(t):    return _c("31", t)
def cyan(t):   return _c("36", t)
def magenta(t):return _c("35", t)

SEVERITY_COLOR = {
    "ok": green,
    "minor": cyan,
    "moderate": yellow,
    "major": red,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _severity_icon(sev: str) -> str:
    return {"ok": "✅", "minor": "🔵", "moderate": "🟡", "major": "🔴"}.get(sev, "❓")


def _print_result_header(result: AgentResult) -> None:
    col = SEVERITY_COLOR.get(result.severity, dim)
    icon = _severity_icon(result.severity)
    elapsed = f"({result.elapsed:.1f}s)"
    print(f"\n{icon}  {bold(result.agent_name)}  "
          f"{col(result.severity.upper())}  {dim(elapsed)}")
    print(f"   {dim(result.summary)}")


def _write_report(results: list[AgentResult], paper_path: str,
                  output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(paper_path).stem if paper_path else "paper"
    out_path = output_dir / f"review_{stem}_{ts}.md"

    lines = [
        f"# Paper Review Report",
        f"",
        f"**Paper:** {paper_path}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        "---",
        "",
    ]

    # Summary table
    lines += [
        "## Summary",
        "",
        "| Agent | Severity | Summary |",
        "|-------|----------|---------|",
    ]
    for r in results:
        lines.append(f"| {r.agent_name} | {_severity_icon(r.severity)} {r.severity} | {r.summary[:80]} |")
    lines += ["", "---", ""]

    # Full findings
    lines.append("## Detailed Findings")
    for r in results:
        lines += [
            "",
            f"## {r.agent_name}",
            f"**Severity:** {_severity_icon(r.severity)} {r.severity}  |  "
            f"**Elapsed:** {r.elapsed:.1f}s",
            "",
            r.findings,
            "",
            "---",
        ]

    # Consolidated new reference suggestions
    all_refs = [item for r in results for item in r.references_found]
    if all_refs:
        lines += ["", "## Suggested New References (Missing Citations)", ""]
        seen: set[str] = set()
        for item in all_refs:
            ref = item.get("ref", {})
            title = ref.get("title", "")
            if title and title not in seen:
                seen.add(title)
                lines.append(
                    f"- **{title}** — {ref.get('authors', '')} "
                    f"({ref.get('year', '')}) *{ref.get('venue', '')}*  "
                    f"→ Para {item.get('para', '?')}: \"{item.get('claim', '')}\"")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-agent scientific paper review pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("paper", nargs="?", help="Path to paper text file (.txt, .md, .tex)")
    parser.add_argument("--code-file", "-c", help="Path to code / data file (enables reproducibility agent)")
    parser.add_argument("--agents", "-a",
                        help="Comma-separated agent IDs to run (default: all). "
                             "Use --list-agents to see IDs.")
    parser.add_argument("--no-scholar", action="store_true",
                        help="Skip agents that query Google Scholar")
    parser.add_argument("--output-dir", "-o", default="reports",
                        help="Directory for report files (default: reports/)")
    parser.add_argument("--list-agents", action="store_true",
                        help="Print all agent IDs and descriptions, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without calling the API")
    parser.add_argument("--model", default=None,
                        help="Override Claude model (default: claude-opus-4-5)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full findings to stdout (not just summary)")
    args = parser.parse_args()

    # --list-agents
    if args.list_agents:
        print(bold("\nAvailable agents:\n"))
        for phase in [1, 2, 3]:
            agents_in_phase = [a for a in ALL_AGENTS if a.priority == phase]
            label = {1: "Phase 1 — independent", 2: "Phase 2 — whole-doc", 3: "Phase 3 — synthesis"}[phase]
            print(bold(f"  {label}"))
            for a in agents_in_phase:
                flags = []
                if a.needs_code:    flags.append("needs --code-file")
                if a.needs_scholar: flags.append("uses Google Scholar")
                flag_str = f"  {dim('(' + ', '.join(flags) + ')')}" if flags else ""
                print(f"    {cyan(a.id):30s}  {a.description}{flag_str}")
            print()
        sys.exit(0)

    if not args.paper:
        parser.print_help()
        sys.exit(1)

    # Load paper
    paper_path = args.paper
    if not Path(paper_path).exists():
        print(red(f"Error: paper file not found: {paper_path}"))
        sys.exit(1)
    paper_text = Path(paper_path).read_text(encoding="utf-8")
    print(bold(f"\n📄 Paper: {paper_path}  ({len(paper_text):,} chars, "
               f"~{len(paper_text.split()):,} words)"))

    # Load code
    code_text = ""
    if args.code_file:
        if not Path(args.code_file).exists():
            print(yellow(f"Warning: code file not found: {args.code_file} — reproducibility agent will be skipped"))
        else:
            code_text = Path(args.code_file).read_text(encoding="utf-8")
            print(bold(f"💻 Code:  {args.code_file}  ({len(code_text):,} chars)"))

    # Select agents
    if args.agents:
        ids = [x.strip() for x in args.agents.split(",")]
        unknown = [i for i in ids if i not in AGENT_REGISTRY]
        if unknown:
            print(red(f"Unknown agent IDs: {', '.join(unknown)}"))
            print("Run --list-agents to see valid IDs.")
            sys.exit(1)
        selected = [AGENT_REGISTRY[i] for i in ids]
    else:
        selected = list(ALL_AGENTS)

    # Filter scholar agents if requested
    if args.no_scholar:
        removed = [a.name for a in selected if a.needs_scholar]
        selected = [a for a in selected if not a.needs_scholar]
        if removed:
            print(yellow(f"⚡ Skipping Scholar agents: {', '.join(removed)}"))

    # Sort by priority
    selected.sort(key=lambda a: a.priority)

    # Override model if requested
    if args.model:
        import agents as agents_module
        agents_module.MODEL = args.model

    print(bold(f"\n🤖 Running {len(selected)} agent(s) in "
               f"{max(a.priority for a in selected)} phase(s):\n"))
    for a in selected:
        scholar_note = "  🔍 Scholar" if a.needs_scholar else ""
        print(f"   Phase {a.priority}  {cyan(a.id):28s}  {dim(a.description)}{scholar_note}")

    if args.dry_run:
        print(yellow("\n[dry-run] No API calls made.\n"))
        sys.exit(0)

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(red("Error: ANTHROPIC_API_KEY environment variable not set."))
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Run agents phase by phase
    results: list[AgentResult] = []
    prior_results: dict[str, AgentResult] = {}
    t_start = time.time()

    current_phase = 0
    for agent in selected:
        if agent.priority != current_phase:
            current_phase = agent.priority
            phase_label = {1: "Phase 1 — independent agents",
                           2: "Phase 2 — whole-document agents",
                           3: "Phase 3 — synthesis"}.get(current_phase, f"Phase {current_phase}")
            print(f"\n{bold('─' * 60)}")
            print(bold(f"  {phase_label}"))
            print(bold('─' * 60))

        print(f"\n  ⏳ Running: {cyan(agent.name)} ...", flush=True)

        ctx = Context(
            paper_text=paper_text,
            code_text=code_text,
            prior_results=prior_results,
            config={},
            client=client,
        )

        try:
            result = agent.run(ctx)
        except Exception as e:
            print(red(f"  ✗ {agent.name} failed: {e}"))
            result = AgentResult(
                agent_id=agent.id, agent_name=agent.name,
                summary=f"Agent error: {e}",
                findings=f"This agent encountered an error: {e}",
                severity="ok", references_found=[], elapsed=0.0,
            )

        results.append(result)
        prior_results[agent.id] = result
        _print_result_header(result)

        if args.verbose and result.findings:
            print()
            for line in result.findings.splitlines()[:30]:
                print(f"      {dim(line)}")
            if result.findings.count('\n') > 30:
                print(dim(f"      ... ({result.findings.count(chr(10))-30} more lines in report)"))

    # Write report
    total = time.time() - t_start
    output_dir = Path(args.output_dir)
    report_path = _write_report(results, paper_path, output_dir)

    print(f"\n{bold('═' * 60)}")
    print(bold("  Review Complete"))
    print(bold('═' * 60))
    print(f"\n  ⏱  Total time: {total:.1f}s")
    print(f"  📋 Agents run: {len(results)}")

    # Severity summary
    from collections import Counter
    sev_counts = Counter(r.severity for r in results)
    parts = []
    for sev in ["major", "moderate", "minor", "ok"]:
        if sev_counts.get(sev, 0):
            col = SEVERITY_COLOR.get(sev, dim)
            parts.append(col(f"{sev_counts[sev]} {sev}"))
    print(f"  {bold('Severity breakdown:')} {', '.join(parts)}")

    # New refs count
    total_new_refs = sum(len(r.references_found) for r in results)
    if total_new_refs:
        print(f"  📚 New reference candidates found: {total_new_refs}")

    print(f"\n  {bold('Report saved:')} {green(str(report_path))}\n")


if __name__ == "__main__":
    main()
