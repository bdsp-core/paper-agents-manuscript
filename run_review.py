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
import json
import os
import sys
import time
from collections import Counter, defaultdict
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
    if result.max_score:
        print(f"   {dim(f'Score: {result.score:.1f}/{result.max_score:.1f} [{result.score_category}]')}")


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _build_repo_snapshot(repo_path: Path, max_total_chars: int = 120_000,
                         max_file_chars: int = 8_000) -> tuple[str, list[str]]:
    allowed_suffixes = {
        ".py", ".r", ".R", ".m", ".jl", ".md", ".txt", ".tex", ".json",
        ".yaml", ".yml", ".toml", ".csv", ".tsv", ".sh"
    }
    skipped_dirs = {".git", "__pycache__", ".pytest_cache", "reports", ".venv", "venv", "node_modules"}
    manifest: list[str] = []
    chunks: list[str] = []
    total_chars = 0

    for path in sorted(repo_path.rglob("*")):
        if any(part in skipped_dirs for part in path.parts):
            continue
        if not path.is_file() or path.suffix not in allowed_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        rel = path.relative_to(repo_path)
        manifest.append(str(rel))
        remaining = max_total_chars - total_chars
        if remaining <= 0:
            continue

        excerpt = text[: min(max_file_chars, remaining)]
        chunks.append(f"\n\n### FILE: {rel}\n{excerpt}")
        total_chars += len(excerpt)

    return "".join(chunks).strip(), manifest


def _score_summary(results: list[AgentResult]) -> dict:
    category_scores: dict[str, float] = defaultdict(float)
    category_max: dict[str, float] = defaultdict(float)
    total_score = 0.0
    total_max = 0.0

    for result in results:
        if result.score is None or result.max_score <= 0:
            continue
        total_score += result.score
        total_max += result.max_score
        category_scores[result.score_category] += result.score
        category_max[result.score_category] += result.max_score

    return {
        "total_score": round(total_score, 1),
        "total_max": round(total_max, 1),
        "percent": round((100.0 * total_score / total_max), 1) if total_max else 0.0,
        "categories": {
            category: {
                "score": round(category_scores[category], 1),
                "max_score": round(category_max[category], 1),
                "percent": round((100.0 * category_scores[category] / category_max[category]), 1)
                if category_max[category] else 0.0,
            }
            for category in sorted(category_scores)
        },
    }


def _write_history(history_path: Path, paper_path: str, report_path: Path,
                   results: list[AgentResult], iteration_label: str) -> Path:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = {}
    else:
        history = {}

    summary = _score_summary(results)
    history.setdefault("paper", paper_path)
    history["updated_at"] = datetime.now().isoformat(timespec="seconds")
    history.setdefault("iterations", [])
    history["iterations"].append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": iteration_label,
        "report_path": str(report_path),
        "total_score": summary["total_score"],
        "total_max": summary["total_max"],
        "percent": summary["percent"],
        "categories": summary["categories"],
        "agents": [
            {
                "id": r.agent_id,
                "name": r.agent_name,
                "severity": r.severity,
                "summary": r.summary,
                "score": r.score,
                "max_score": r.max_score,
                "category": r.score_category,
            }
            for r in results
        ],
    })
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history_path


def _render_score_bar(score: float, max_score: float) -> str:
    pct = 0.0 if max_score == 0 else (100.0 * score / max_score)
    return f"<div class='bar'><span style='width:{pct:.1f}%'></span></div>"


def _build_svg_chart(iterations: list[dict], width: int = 760, height: int = 260) -> str:
    if not iterations:
        return ""
    padding = 30
    usable_w = width - 2 * padding
    usable_h = height - 2 * padding
    x_positions = [padding + usable_w * i / max(1, len(iterations) - 1) for i in range(len(iterations))]
    points = []
    for x, iteration in zip(x_positions, iterations):
        y = padding + usable_h * (1 - (iteration["percent"] / 100.0))
        points.append((x, y))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    grid_lines = []
    for marker in [0, 25, 50, 75, 100]:
        y = padding + usable_h * (1 - marker / 100.0)
        grid_lines.append(f"<line x1='{padding}' y1='{y:.1f}' x2='{width-padding}' y2='{y:.1f}' class='grid-line' />")
        grid_lines.append(f"<text x='8' y='{y+4:.1f}' class='axis'>{marker}</text>")

    circles = []
    labels = []
    for (x, y), iteration in zip(points, iterations):
        circles.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' class='pt' />")
        labels.append(f"<text x='{x:.1f}' y='{height-8}' text-anchor='middle' class='axis'>{iteration['label']}</text>")

    return (
        f"<svg viewBox='0 0 {width} {height}' class='chart'>"
        + "".join(grid_lines)
        + f"<polyline points='{polyline}' class='line' />"
        + "".join(circles)
        + "".join(labels)
        + "</svg>"
    )


def _write_dashboard(dashboard_path: Path, history_path: Path) -> Path:
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        history = {"iterations": []}

    iterations = history.get("iterations", [])
    latest = iterations[-1] if iterations else {}
    chart = _build_svg_chart(iterations[-12:])

    category_rows = []
    for category, values in latest.get("categories", {}).items():
        category_rows.append(
            "<tr>"
            f"<td>{category}</td>"
            f"<td>{values['score']:.1f} / {values['max_score']:.1f}</td>"
            f"<td>{values['percent']:.1f}%</td>"
            f"<td>{_render_score_bar(values['score'], values['max_score'])}</td>"
            "</tr>"
        )

    agent_rows = []
    for agent in latest.get("agents", []):
        agent_rows.append(
            "<tr>"
            f"<td>{agent['name']}</td>"
            f"<td>{agent['category']}</td>"
            f"<td>{agent['score']:.1f} / {agent['max_score']:.1f}</td>"
            f"<td>{agent['severity']}</td>"
            f"<td>{agent['summary']}</td>"
            "</tr>"
        )

    history_rows = []
    for iteration in reversed(iterations[-12:]):
        history_rows.append(
            "<tr>"
            f"<td>{iteration['label']}</td>"
            f"<td>{iteration['timestamp']}</td>"
            f"<td>{iteration['total_score']:.1f} / {iteration['total_max']:.1f}</td>"
            f"<td>{iteration['percent']:.1f}%</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="30">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paper Agent Scoreboard</title>
  <style>
    :root {{
      --bg: #f6f4ef;
      --panel: #fffdf8;
      --ink: #1c1c18;
      --muted: #6b6a63;
      --accent: #1f6f78;
      --line: #d8d3c7;
      --good: #3c8d5a;
    }}
    body {{ margin: 0; font-family: Georgia, "Times New Roman", serif; background: linear-gradient(180deg, #efe8da, var(--bg)); color: var(--ink); }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .hero, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 22px; box-shadow: 0 12px 30px rgba(0,0,0,0.05); }}
    .meta {{ color: var(--muted); margin-top: 8px; }}
    .score {{ font-size: 3rem; color: var(--accent); margin: 8px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; margin-top: 20px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.96rem; }}
    th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .bar {{ width: 100%; height: 10px; border-radius: 999px; background: #eee7da; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--good)); }}
    .chart {{ width: 100%; height: auto; }}
    .grid-line {{ stroke: #d8d3c7; stroke-width: 1; }}
    .line {{ fill: none; stroke: var(--accent); stroke-width: 4; }}
    .pt {{ fill: var(--good); }}
    .axis {{ fill: var(--muted); font-size: 11px; }}
    code {{ background: #f1ebde; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Paper Agent Scoreboard</h1>
      <div class="score">{latest.get("total_score", 0):.1f} / {latest.get("total_max", 0):.1f} ({latest.get("percent", 0):.1f}%)</div>
      <div class="meta">Auto-refreshes every 30 seconds. Source data: <code>{history_path.name}</code>. Latest iteration: {latest.get("label", "n/a")}.</div>
      <div class="meta">Paper: {history.get("paper", "n/a")}</div>
    </section>
    <div class="grid">
      <section class="panel">
        <h2>Total Score Trend</h2>
        {chart}
      </section>
      <section class="panel">
        <h2>Category Scores</h2>
        <table>
          <thead><tr><th>Category</th><th>Score</th><th>%</th><th>Bar</th></tr></thead>
          <tbody>{''.join(category_rows)}</tbody>
        </table>
      </section>
    </div>
    <div class="grid">
      <section class="panel">
        <h2>Agent Scores</h2>
        <table>
          <thead><tr><th>Agent</th><th>Category</th><th>Score</th><th>Severity</th><th>Summary</th></tr></thead>
          <tbody>{''.join(agent_rows)}</tbody>
        </table>
      </section>
      <section class="panel">
        <h2>Iteration History</h2>
        <table>
          <thead><tr><th>Label</th><th>Timestamp</th><th>Total</th><th>%</th></tr></thead>
          <tbody>{''.join(history_rows)}</tbody>
        </table>
      </section>
    </div>
  </main>
</body>
</html>
"""
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path


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

    score_summary = _score_summary(results)
    lines += [
        "## Score Summary",
        "",
        f"**Total score:** {score_summary['total_score']:.1f} / {score_summary['total_max']:.1f} ({score_summary['percent']:.1f}%)",
        "",
        "| Category | Score | Percent |",
        "|----------|-------|---------|",
    ]
    for category, values in score_summary["categories"].items():
        lines.append(
            f"| {category} | {values['score']:.1f} / {values['max_score']:.1f} | {values['percent']:.1f}% |"
        )
    lines += ["", "---", ""]

    # Summary table
    lines += [
        "## Summary",
        "",
        "| Agent | Category | Score | Severity | Summary |",
        "|-------|----------|-------|----------|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r.agent_name} | {r.score_category} | {r.score:.1f} / {r.max_score:.1f} | "
            f"{_severity_icon(r.severity)} {r.severity} | {r.summary[:80]} |"
        )
    lines += ["", "---", ""]

    # Full findings
    lines.append("## Detailed Findings")
    for r in results:
        lines += [
            "",
            f"## {r.agent_name}",
            f"**Score:** {r.score:.1f} / {r.max_score:.1f} ({r.score_category})",
            "",
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
    parser.add_argument("--repo-path", default=".",
                        help="Repository path to snapshot for reproducibility/truthfulness checks (default: current directory)")
    parser.add_argument("--history-file", default=None,
                        help="Path to score history JSON (default: <output-dir>/score_history.json)")
    parser.add_argument("--dashboard-file", default=None,
                        help="Path to HTML dashboard (default: <output-dir>/score_dashboard.html)")
    parser.add_argument("--iteration-label", default=None,
                        help="Optional label for this review iteration in the score history")
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
    config = _load_config(Path("review_config.json"))
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

    repo_path = Path(args.repo_path).resolve()
    repo_snapshot, repo_manifest = _build_repo_snapshot(repo_path)
    if repo_manifest:
        print(bold(f"🧭 Repo snapshot: {repo_path}  ({len(repo_manifest)} files indexed)"))

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
            repository_text=repo_snapshot,
            repository_manifest=repo_manifest,
            prior_results=prior_results,
            config=config,
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
                severity="major", references_found=[], elapsed=0.0,
                score=0.0, max_score=agent.max_score, score_category=agent.score_category,
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

    score_summary = _score_summary(results)
    total_score_text = f"{score_summary['total_score']:.1f}/{score_summary['total_max']:.1f}"
    print(f"  {bold('Score:')} {green(total_score_text)} ({score_summary['percent']:.1f}%)")
    for category, values in score_summary["categories"].items():
        print(f"     {category:14s} {values['score']:.1f}/{values['max_score']:.1f} ({values['percent']:.1f}%)")

    history_path = Path(args.history_file) if args.history_file else output_dir / "score_history.json"
    dashboard_path = Path(args.dashboard_file) if args.dashboard_file else output_dir / "score_dashboard.html"
    existing_iterations = 0
    if history_path.exists():
        try:
            existing_iterations = len(json.loads(history_path.read_text(encoding="utf-8")).get("iterations", []))
        except json.JSONDecodeError:
            existing_iterations = 0
    iteration_label = args.iteration_label or f"iter-{existing_iterations + 1}"
    history_path = _write_history(history_path, paper_path, report_path, results, iteration_label)
    dashboard_path = _write_dashboard(dashboard_path, history_path)

    print(f"\n  {bold('Report saved:')} {green(str(report_path))}")
    print(f"  {bold('Score history:')} {green(str(history_path))}")
    print(f"  {bold('Dashboard:')} {green(str(dashboard_path))}\n")


if __name__ == "__main__":
    main()
