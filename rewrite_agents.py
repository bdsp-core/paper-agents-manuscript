"""
rewrite_agents.py — the "Clarity and Grace" REWRITE agents.

Unlike the agents in agents.py (which *review* and score a manuscript), these three
*rewrite* it, one paragraph at a time, applying Williams' *Style: Lessons in Clarity
and Grace*:

    clarity   -> characters as subjects, actions as verbs, kill fixable
                 nominalizations, active voice (deferring to flow)
    cohesion  -> old-before-new (topic position), new/important-at-end (stress
                 position), consistent topic strings
    concision -> cut redundancy/filler/metadiscourse, control sprawl

They are run by run_rewrite.py in that order, with a fidelity gate after each
paragraph (see agents.fidelity_verify). Each agent returns ONLY the revised
paragraph text — no change log, no preamble — so the output can be reassembled
directly into a manuscript.

The full source-of-truth prompts live in clarity_and_grace/*.md; the SYSTEM strings
below are the ported, condensed versions.
"""

from __future__ import annotations

import textwrap

from agents import _call, _preservation_block

_RETURN_RULE = (
    "\n\n## Output\nReturn ONLY the revised paragraph, ready to drop into the "
    "manuscript. No preamble, no quotes, no change log, no commentary. If the "
    "paragraph is already clear and needs no change, return it unchanged."
)


class RewriteAgent:
    id: str = ""
    name: str = ""
    SYSTEM: str = ""
    max_tokens: int = 1500

    def rewrite(self, client, paragraph: str, config: dict) -> str:
        system = self.SYSTEM + _preservation_block(config) + _RETURN_RULE
        out = _call(client, system, f"PARAGRAPH:\n\n{paragraph}",
                    max_tokens=self.max_tokens)
        return out.strip()


class ClarityRewriter(RewriteAgent):
    id = "clarity"
    name = "Clarity (rewrite)"
    SYSTEM = textwrap.dedent("""
        You are the CLARITY agent. Rewrite the paragraph so a reader instantly sees
        WHO is doing WHAT. Change wording only — never claims, numbers, citations, or
        technical terms.

        For each sentence: find the real CHARACTER (the actor) and the real ACTION,
        then make the character the grammatical subject and the action the verb.

        Kill fixable NOMINALIZATIONS — actions/qualities frozen into nouns propped up
        by an empty verb (is/was/has/makes/conducts/performs/provides):
        - "We performed an evaluation of the model." -> "We evaluated the model."
        - "There was a reduction in seizure burden." -> "Seizure burden fell."
        - "the development of a method for the detection of X" -> "to detect X"
        BUT KEEP nominalizations that are topic terms / defined concepts (see the
        PRESERVATION RULES below) or that point back to old information as cohesive
        glue ("these findings", "this decision").

        Prefer active voice, but KEEP a passive when it puts old/familiar information
        first — the cohesion pass owns that call. Keep subject and verb close together.
    """).strip()


class CohesionRewriter(RewriteAgent):
    id = "cohesion"
    name = "Cohesion (rewrite)"
    SYSTEM = textwrap.dedent("""
        You are the COHESION agent. Clarity makes one sentence clear; you make the
        paragraph FLOW. Reorder and re-hinge information; do not invent or delete content.

        - Topic position: each sentence should open on familiar/old information — a
          term from the previous sentence or a previously named entity. New material
          should not lead. A passive that puts the old topic in front is correct.
        - Stress position: push the newest, heaviest, or most important unit to the
          END of the sentence, where the reader expects the payoff.
        - Consistent topics: keep the grammatical subjects pointing at the same small
          set of topics through the paragraph; stitch each sentence's end-idea to the
          next sentence's opening (old-to-new).

        Reordering must not change which claim the paragraph emphasizes.
    """).strip()


class ConcisionRewriter(RewriteAgent):
    id = "concision"
    name = "Concision (rewrite)"
    SYSTEM = textwrap.dedent("""
        You are the CONCISION agent. Make every sentence as short as it can be WITHOUT
        losing meaning, hedging, or emphasis.

        Cut, in this order: redundant pairs ("full and complete"); redundant modifiers
        ("future plans", "completely eliminate"); redundant categories ("period of
        time" -> "period"); filler ("basically", "actually", "in order to" -> "to");
        metadiscourse ("it is important to note that"); "the fact that" ("due to the
        fact that" -> "because"); restating the prior sentence; prepositional-phrase
        pileups. Break run-on sentences carrying more than one main idea; one main
        idea per sentence by default.

        CRITICAL: scientific concision is NOT compression. NEVER cut a hedge or a scope
        condition even when it looks like filler ("may be associated with", "in the
        external validation set", "under these assumptions"). See PRESERVATION RULES.
    """).strip()


# Ordered pipeline used by run_rewrite.py
PIPELINE = [ClarityRewriter(), CohesionRewriter(), ConcisionRewriter()]
