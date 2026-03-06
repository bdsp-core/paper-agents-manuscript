# paper-review-agents

Multi-agent pipeline for reviewing scientific manuscript drafts before journal submission.
Each agent targets a specific dimension of writing quality, grounded in established frameworks
(VSNC/Winston, Gopen & Swan, Adelson/Freeman, Strunk & White). Two agents query Google Scholar
to surface missing citations and verify existing ones.

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
| 9 | 1 | **Reproducibility Check** | Results traceable to code/data · Methods match implementation *(requires --code-file)* |
| 10 | 2 | **Internal Consistency** | Terminology · Numbers · Claims across abstract/methods/results/discussion |
| 11 | 2 | **Discussion & Related Work** | Positioning in literature · Gaps · Limitations · Strength of conclusion |
| 12 | 2 | **Missing References** 🔍 | Paragraph-by-paragraph scan for uncited claims → Google Scholar search for candidates |
| 13 | 2 | **Reference Quality** 🔍 | Verifies cited refs via Scholar · Flags wrong papers · Suggests better canonical refs |
| 14 | 3 | **Synthesis & Action Plan** | Cross-cutting synthesis of all agents → ranked top-10 action plan + readiness verdict |

🔍 = queries Google Scholar (can be skipped with `--no-scholar`)

## Quick start

```bash
pip install anthropic scholarly
export ANTHROPIC_API_KEY=sk-ant-...

# Full review
python run_review.py my_paper.txt

# With code for reproducibility checking
python run_review.py my_paper.txt --code-file analysis.py

# Skip Scholar agents (faster, no rate-limit risk)
python run_review.py my_paper.txt --no-scholar

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

## Pipeline architecture

Agents run in three sequential phases. Each phase can use the outputs of prior phases.

```
Phase 1 (independent)          Phase 2 (whole-doc)      Phase 3 (synthesis)
──────────────────────         ────────────────────      ───────────────────
vsnc                 ─┐
intro                 │
sentences             │──→  consistency          ─┐
voice                 │     discussion             │──→  orchestrator
conciseness           │     missing_refs  🔍       │     (action plan)
paragraphs            │     ref_quality   🔍      ─┘
acronyms              │
figures_tables        │
reproducibility      ─┘
```

## Input formats

The paper file can be plain text (`.txt`), Markdown (`.md`), or LaTeX (`.tex`).
For LaTeX, the agents work on the raw source — LaTeX commands are visible but
Claude handles them well in practice. For best results, strip comments first.

The code file (`--code-file`) can be Python, R, MATLAB, or a plain-text description
of the computational pipeline. Multiple files can be concatenated.

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
    "citation_style": "numbered superscript"
}
```

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

## License

MIT

## About

Built for the BDSP Computational Clinical Neurology Lab at Massachusetts General Hospital / Harvard Medical School.
