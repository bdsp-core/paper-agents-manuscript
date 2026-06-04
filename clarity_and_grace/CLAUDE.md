# Paper Restyle — orchestration

This repo restyles a paper at the **sentence and paragraph level** using
Williams' *Style: Lessons in Clarity and Grace*, decomposed into four subagents
in `.claude/agents/`. This file is the coordinator the main session follows.

## Global invariants (apply to every agent, every paragraph)

Style edits change *wording and order only*. They must NEVER change:

- the **meaning** or **direction** of any claim;
- **hedge strength / calibration** — `may, might, could, appears, suggests, is
  associated with, consistent with, under assumption X, in cohort Y` must
  survive (reworded is fine, removed/strengthened is not);
- **numbers, statistics, units, dates, thresholds** (n, %, p, CI, AUC, …);
- **citations / cross-refs / math**: `\cite{} \ref{} \label{} \eqref{} $…$`,
  equation environments, footnote markers;
- **defined terms, acronyms, and topic nouns** — these are often
  nominalizations (*delirium, classification, inference, phenotype,
  calibration*) and are KEPT, not "fixed";
- **LaTeX / Markdown markup** — output must still compile.

## Pipeline (run per paragraph, in this order)

Order is deliberate: clarity changes the words, cohesion changes their order,
concision trims what's left, fidelity verifies nothing broke.

1. **clarity** → characters→subjects, actions→verbs, kill fixable
   nominalizations, active voice (deferring to flow), subject–verb proximity.
2. **cohesion** → old-before-new (topic position), new/important-at-end (stress
   position), consistent topic strings.
3. **concision** → cut redundancy/filler/metadiscourse, control sprawl —
   preserving hedges and scope.
4. **fidelity-check** → PASS/FLAG. On FLAG, return the paragraph to the named
   agent with the preserve-instruction, then re-run fidelity. Max 2 retries; if
   still flagged, leave the original paragraph and report it for human review.

## Trade-off priority (when principles collide)

clarity of who-does-what  >  cohesion (old→new flow)  >  emphasis (stress)  >
concision. Active voice yields to cohesion: keep a passive when it puts old/short
information in topic position. Never sacrifice an invariant for any principle.

## Modes

- **`diagnose`** (default first run): produce the change log + flags for the
  whole document, **edit nothing**. Let the human review before applying.
- **`apply`**: write edits in place after `cp paper.tex paper.tex.bak`, and emit
  a unified diff (`git diff` or `diff -u`) so every change is reviewable.

## Running it

Process the document paragraph by paragraph (skip floats: figures, tables,
equations, code listings, abstract-only fields if asked). For each paragraph,
invoke the four agents in order and collect their change logs. After the
document, produce a summary: paragraphs touched, nominalizations removed,
passives kept-for-flow, and every fidelity FLAG that needed human review.

Principle 10 ("write for the reader") is not a step — it is the test each agent
applies: would a busy, expert reader grasp this faster now, without losing any
precision? If an edit makes prose snappier but a domain reader slower or less
sure, revert it.
