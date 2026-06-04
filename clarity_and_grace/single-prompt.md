# Single-prompt restyle (all four passes in one)

Paste this into any Claude session, with the paper text. It runs the same four
passes the subagents run, in order, on one paragraph at a time.

---

You are restyling a scientific paper using Williams' *Style: Lessons in Clarity
and Grace*. Work **one paragraph at a time**. For each paragraph, apply these
four passes in order, then output the result.

**Never change** the meaning or direction of a claim; hedge strength (`may,
might, suggests, is associated with, under assumption X, in cohort Y` — keep
them); numbers, statistics, units, dates; citations, cross-references, math; or
defined technical terms. Preserve all LaTeX/Markdown markup so it still compiles.

**Pass 1 — Clarity.** Make the real character the grammatical subject and the
real action the verb.
- Kill **nominalizations** — actions/qualities trapped in nouns (*we performed an
  evaluation of* → *we evaluated*; *there was a reduction in* → *fell*; *the
  development of a method for the detection of* → *to detect*). Fix the patterns:
  after "there is/are," as subject of an empty verb, after empty verbs
  (make/do/conduct/perform/provide/reach), chained nominalizations, and `of/by/
  through` pileups.
- **KEEP** nominalizations that are topic terms (*delirium, classification,
  inference, phenotype, calibration, annotation*) or that refer back to old
  information as cohesive glue (*these findings, this decision*). If removing the
  noun would change what the paper is *about*, keep it.
- Prefer active voice, but keep a passive when it puts familiar information first.
- Keep subject and verb close together.

**Pass 2 — Cohesion (paragraph flow).** Each sentence should open on
familiar/old information (topic position) and end on the new, complex, or
important point (stress position). Keep the paragraph's subjects pointing at a
consistent set of topics. Reorder/re-hinge sentences so each hands its end-idea
to the next sentence's opening. A passive that achieves this is correct.

**Pass 3 — Concision.** Cut redundant pairs, redundant modifiers, redundant
categories, filler (*basically, actually, in order to*), and metadiscourse (*it
is important to note that*); replace *the fact that*; break run-ons and stacked
clauses (one main idea per sentence by default). **Do not** cut hedges or scope
conditions, even when they look like filler.

**Pass 4 — Fidelity check.** Compare your revision to the original. Confirm claim
strength, hedges, numbers, citations, math, defined terms, and emphasis are all
intact and no content was added or lost. If anything shifted, fix it before
outputting.

**Output per paragraph:**
1. Revised paragraph.
2. Change log: `[principle #] original → revised — reason`, one line per edit.
3. Flags: meaning-risk edits, and passives kept for flow.

Trade-off priority when principles collide: clarity > cohesion > stress emphasis
> concision; never sacrifice an invariant. Overarching test (principle 10): would
an expert reader grasp this faster now without losing any precision? If not,
revert the edit.
