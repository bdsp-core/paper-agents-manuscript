# paper-review-agents

Multi-agent pipeline for **reviewing and rewriting** scientific manuscript drafts before
journal submission. Each agent targets a specific dimension of writing quality, grounded in
established frameworks (VSNC/Winston, Gopen & Swan, Adelson/Freeman, Strunk & White, and
Williams' *Style: Lessons in Clarity and Grace*). Two agents query Google Scholar to surface
missing citations and verify existing ones.

The tool has two halves that form a closed loop:

- **Review** (`run_review.py`) — 16 agents grade the manuscript across argument, prose, rigor,
  citations, fidelity, and more, producing a scored report + action plan.
- **Rewrite** (`run_rewrite.py`) — the *Clarity and Grace* pipeline (clarity → cohesion →
  concision) improves the prose paragraph by paragraph, with a **fidelity gate** that reverts
  any edit which would change a claim, hedge, number, or citation. See
  [Rewrite mode](#rewrite-mode-clarity-and-grace).

The pipeline also supports iterative score-tracking, a repository-grounded truthfulness
check, and an auto-refreshing HTML dashboard so you can see whether rewrites are improving
the manuscript over time — and the rewrite half can re-score itself into that dashboard
automatically.

## Agents

| # | Phase | Agent | What it checks |
|---|-------|-------|----------------|
| 1 | 1 | **VSNC Framework** | Vision · Steps · News · Contributions · 5 S's (slogan, symbol, salient, surprise, story) |
| 2 | 1 | **Introduction Audit** | Adelson formula · Kajiya "dynamite intro" · Freeman tone |
| 3 | 1 | **Sentence Architecture** | Gopen & Swan: stress positions, topic positions, subject-verb proximity, nominalizations |
| 4 | 1 | **Voice & Tense** | Active/passive ratio · Past for methods/results · Present for facts · Tense shifts |
| 5 | 1 | **Conciseness Audit** | Wordy phrases · Nominalizations · Redundancy · Throat-clearing openers |
| 6 | 1 | **Paragraph Quality** | Topic sentences · Unity · Logical flow · Reader-first principle (Knuth) |
| 7 | 1 | **Acronym Audit** | Every acronym defined before first use · Double-definitions · Post-definition consistency |
| 8 | 1 | **Figures, Tables & Captions** | Coverage · Caption titles state findings · Panel descriptions · Self-contained captions |
| 9 | 1 | **Reproducibility Check** | Results traceable to code/data or repository snapshot · Methods match implementation |
| 10 | 2 | **Truthfulness & Code Grounding** | Checks whether claims are derivable from repository code and honestly described |
| 11 | 2 | **Internal Consistency** | Terminology · Numbers · Claims across abstract/methods/results/discussion |
| 12 | 2 | **Discussion & Related Work** | Positioning in literature · Gaps · Limitations · Strength of conclusion |
| 13 | 2 | **Fidelity / Meaning-Preservation** | Verifies a rewrite kept every claim, hedge, number, and citation (diff mode via `--revised`) |
| 14 | 2 | **Missing References** 🔍 | Paragraph-by-paragraph scan for uncited claims → Google Scholar search for candidates |
| 15 | 2 | **Reference Quality** 🔍 | Verifies cited refs via Scholar · Flags wrong papers · Suggests better canonical refs |
| 16 | 3 | **Synthesis & Action Plan** | Cross-cutting synthesis of all agents → ranked top-10 action plan + readiness verdict |

🔍 = queries Google Scholar (can be skipped with `--no-scholar`)

## Rewrite mode (Clarity and Grace)

The agents above *grade* the manuscript. The companion **rewrite pipeline** (`run_rewrite.py`)
*improves* it, applying Williams' *Style: Lessons in Clarity and Grace* one paragraph at a time:

```
clarity  →  cohesion  →  concision  →  fidelity gate (revert on FLAG)
```

- **clarity** — characters as subjects, actions as verbs, kill fixable nominalizations (but
  KEEP topic-term nominalizations like *calibration*, *phenotype*), active voice deferring to flow
- **cohesion** — old-before-new (topic position), important-at-end (stress position), consistent topics
- **concision** — cut filler/redundancy/metadiscourse, control sprawl — **without** touching hedges or scope
- **fidelity gate** — after each paragraph, [agents.fidelity_verify](agents.py) checks that every claim,
  hedge, number, and citation survived; anything risky is **reverted to the original**

```bash
# Dry run: which paragraphs would be rewritten (no API calls)
python run_rewrite.py paper.md --dry-run

# Rewrite -> paper.revised.md  (+ paper.revised.changelog.md), fidelity gate on
python run_rewrite.py paper.md

# Rewrite, then score the result against the original — the delta lands in the dashboard
python run_rewrite.py paper.md --review --iteration-label rewrite-1
```

This closes the scoring loop: instead of rewriting by hand between iterations, run `run_rewrite.py`
and the new version is produced, fidelity-checked, and re-scored automatically. The agent prompts
live in [clarity_and_grace/](clarity_and_grace/) (also exposed as Claude Code subagents in
[.claude/agents/](.claude/agents/)).

### Verifying a rewrite you already have

Even without the rewrite pipeline, you can fidelity-check any two versions:

```bash
python run_review.py revised.md --revised original.md --agents fidelity
```

## Quick start

```bash
pip install anthropic scholarly
export ANTHROPIC_API_KEY=sk-ant-...

# Full review
python run_review.py my_paper.txt

# With code for reproducibility checking
python run_review.py my_paper.txt --code-file analysis.py

# Ground the review in the repository codebase
python run_review.py my_paper.txt --repo-path .

# Skip Scholar agents (faster, no rate-limit risk)
python run_review.py my_paper.txt --no-scholar

# Track iterations for score improvement
python run_review.py my_paper.txt --iteration-label baseline
python run_review.py my_paper.txt --iteration-label rewrite-1

# Run only specific agents
python run_review.py my_paper.txt --agents vsnc,intro,paragraphs,discussion

# See all agent IDs
python run_review.py --list-agents

# Dry run (show plan without calling API)
python run_review.py my_paper.txt --dry-run

# Print findings to stdout as well as report file
python run_review.py my_paper.txt --verbose
```

Reports are saved to `reports/review_<papername>_YYYYMMDD_HHMMSS.md`.
Score history is saved to `reports/score_history.json`.
The dashboard is saved to `reports/score_dashboard.html` and auto-refreshes every 30 seconds.

## Scoring loop

Each agent now contributes points to a tracked score, grouped into categories such as
`argument`, `prose`, `rigor`, `citations`, and `truthfulness`.

This makes it easy to run an AutoResearch-style loop:
1. Evaluate the manuscript.
2. Inspect the weakest categories or agents.
3. Rewrite the paper to target those weaknesses.
4. Re-run with a new `--iteration-label`.
5. Watch total and category scores improve in the dashboard.

## Pipeline architecture

Agents run in three sequential phases. Each phase can use the outputs of prior phases.

```
Phase 1 (independent)          Phase 2 (whole-doc)      Phase 3 (synthesis)
──────────────────────         ────────────────────      ───────────────────
vsnc                 ─┐
intro                 │
sentences             │──→  consistency          ─┐
voice                 │     discussion             │──→  orchestrator
conciseness           │     fidelity  (--revised)  │     (action plan)
paragraphs            │     missing_refs  🔍        │
acronyms              │     ref_quality   🔍       ─┘
figures_tables        │
reproducibility      ─┘     truthfulness

  run_rewrite.py:  clarity → cohesion → concision → fidelity gate  (rewrite half of the loop)
```

## Input formats

The paper file can be plain text (`.txt`), Markdown (`.md`), or LaTeX (`.tex`).
For LaTeX, the agents work on the raw source — LaTeX commands are visible but
Claude handles them well in practice. For best results, strip comments first.

The code file (`--code-file`) can be Python, R, MATLAB, or a plain-text description
of the computational pipeline. Multiple files can be concatenated.

The repository snapshot (`--repo-path`) is used by the reproducibility and truthfulness
agents to compare manuscript claims against the code that is actually present in the repo.

## Google Scholar notes

`scholarly` queries the public Google Scholar interface without an API key.
Scholar occasionally rate-limits or CAPTCHAs aggressive crawlers.
If you hit errors:
- Add `--no-scholar` to skip both reference agents
- Or install `scholarly` with a proxy: `pip install scholarly[proxy]`
- For production use, consider [SerpAPI](https://serpapi.com) (paid, reliable)

## Configuring for your journal

Edit `review_config.json`:

```json
{
    "journal_name": "Nature Medicine",
    "word_limit": 3000,
    "abstract_word_limit": 150,
    "max_figures": 4,
    "citation_style": "numbered superscript",

    "protected_terms": ["calibration", "phenotype", "empirical null"],
    "protect_hedges": true
}
```

`protected_terms` and `protect_hedges` feed the **preservation rules** that the prose
review agents and the rewrite pipeline obey. They stop the Strunk & White / Gopen & Swan
agents from "fixing" a topic-term nominalization (*calibration* → *calibrate*) or cutting a
scientific hedge (*may be associated with*) as if it were clutter — the failure mode that
makes naive style tools dangerous on a clinical manuscript.

## Requirements

- Python 3.10+
- `anthropic` Python SDK
- `scholarly` (for reference agents)
- `ANTHROPIC_API_KEY` environment variable

## Companion: Figure Review Agents

This tool focuses on **manuscript text** review. For automated review of scientific
**figures** (layout, color, accessibility, Tufte principles), see the companion repo:

**[paper-agents-figures](https://github.com/bdsp-core/paper-agents-figures.git)**

The two tools are designed to work together for a complete pre-submission check:
1. Run `paper-agents-figures` to review all figures
2. Run `paper-agents-manuscript` (this repo) to review the full manuscript text

## Files

| File | Role |
|------|------|
| `run_review.py` | CLI for the review pipeline (scoring, dashboard, `--revised` fidelity diff) |
| `run_rewrite.py` | CLI for the Clarity-and-Grace rewrite pipeline (`--review` closes the loop) |
| `agents.py` | The 16 review agents, the preservation rules, and `fidelity_verify()` |
| `rewrite_agents.py` | The clarity / cohesion / concision rewriters |
| `review_config.json` | Journal settings + `protected_terms` / `protect_hedges` |
| `clarity_and_grace/` | Source-of-truth prompts for the rewrite agents (+ Williams coordinator) |
| `.claude/agents/` | The same four agents exposed as Claude Code subagents |

## License

MIT

## About

Built for the BDSP Computational Clinical Neurology Lab at Massachusetts General Hospital / Harvard Medical School.

The Clarity-and-Grace rewrite pipeline and fidelity gate apply Joseph M. Williams'
*Style: Lessons in Clarity and Grace*, adapted so that scientific hedges, statistics,
citations, and topic terms are preserved exactly while prose is sharpened.
