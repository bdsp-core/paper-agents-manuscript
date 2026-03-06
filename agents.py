"""
agents.py — All agent definitions for the paper-review pipeline.

Each agent is a dataclass with:
  - id / name / description
  - priority  (1 = runs first/independently, 2 = needs prior context, 3 = final synthesis)
  - needs_code (bool)  — whether code/data context is required
  - needs_scholar (bool) — whether this agent does Google Scholar look-ups
  - run(ctx) -> AgentResult  — the main entry point

The Context object (defined in run_review.py) carries:
  - paper_text: full paper as a string
  - code_text:  optional code / data descriptions
  - figures:    list of figure paths (optional)
  - prior_results: dict[agent_id -> AgentResult] from earlier phases
  - config:     journal_config dict
  - client:     anthropic.Anthropic instance
"""

from __future__ import annotations

import re
import time
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import anthropic


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    agent_id: str
    agent_name: str
    summary: str            # One-line verdict, e.g. "3 undefined acronyms found"
    findings: str           # Full markdown body
    severity: str           # "ok" | "minor" | "moderate" | "major"
    references_found: list  # Populated only by reference agents
    elapsed: float = 0.0


@dataclass
class Context:
    paper_text: str
    code_text: str = ""
    figures: list = field(default_factory=list)
    prior_results: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    client: Optional[anthropic.Anthropic] = None


# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-5"


def _call(client: anthropic.Anthropic, system: str, user: str,
          max_tokens: int = 2048) -> str:
    """Single Claude API call; returns text."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def _parse_severity(text: str) -> str:
    """Heuristic: look for explicit severity tag the agent is asked to emit."""
    text_lower = text.lower()
    if "severity: major" in text_lower or "🔴" in text:
        return "major"
    if "severity: moderate" in text_lower or "🟡" in text:
        return "moderate"
    if "severity: minor" in text_lower or "🟢" in text:
        return "minor"
    return "ok"


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:120]
    return "(no summary)"


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------

class BaseAgent:
    id: str = ""
    name: str = ""
    description: str = ""
    priority: int = 1        # 1, 2, or 3
    needs_code: bool = False
    needs_scholar: bool = False

    def run(self, ctx: Context) -> AgentResult:
        raise NotImplementedError


# ===========================================================================
#  PHASE 1 — independent per-document agents
# ===========================================================================

class VSNCAgent(BaseAgent):
    id = "vsnc"
    name = "VSNC Framework"
    description = "Vision · Steps · News · Contributions · 5 S's (Patrick Winston/MIT)"
    priority = 1

    SYSTEM = textwrap.dedent("""
        You are an expert scientific writing reviewer applying the VSNC framework
        (Patrick Winston, MIT) and the 5 S's paradigm.

        ## VSNC Framework (evaluate abstract AND introduction)
        - Vision: Is the big idea stated explicitly? What concrete advances does it enable?
          Is there an "empowerment promise" — does the reader know what they'll gain?
        - Steps: Are the concrete steps needed to execute the idea enumerated?
        - News: Are specific results listed with maximum specificity? (numbers, benchmarks)
        - Contributions: Are contributions stated using strong sanctioned verbs —
          *prove, demonstrate, implement, test, frame, survey, identify, present, show*?

        ## 5 S's (memorability)
        - Slogan: Is there a repeated phrase that anchors the paper in the reader's mind?
        - Symbol: Is there a repeated figure/visual that embodies the main idea?
        - Salient: Is there ONE standout idea? Too many competing ideas means none sticks.
        - Surprise: Is there something unexpected that hooks the reader?
        - Story: Is there a narrative arc — problem, journey, resolution?

        ## Inversion Heuristic (Winston)
        Put yourself in the reader's shoes. Does the abstract read cold?
        Does skimming topic sentences give the paper's full argument?

        ## Output format
        Start with a one-line SUMMARY and SEVERITY (ok / minor / moderate / major).
        Then grade each VSNC component and each S with ✅ / ⚠️ / ❌.
        Quote the paper and suggest concrete text to add where missing.
    """).strip()

    def run(self, ctx: Context) -> AgentResult:
        t0 = time.time()
        findings = _call(ctx.client, self.SYSTEM,
                         f"PAPER:\n\n{ctx.paper_text}")
        return AgentResult(
            agent_id=self.id, agent_name=self.name,
            summary=_first_line(findings), findings=findings,
            severity=_parse_severity(findings), references_found=[],
            elapsed=time.time() - t0,
        )


class IntroductionAgent(BaseAgent):
    id = "intro"
    name = "Introduction Audit"
    description = "Adelson formula · Kajiya 'dynamite intro' · Freeman tone"
    priority = 1

    SYSTEM = textwrap.dedent("""
        You are a scientific writing reviewer applying the Freeman/Adelson/Kajiya framework.

        ## The Adelson Formula — grade each step A–F
        1. Problem stated clearly and early?
        2. Reader told WHY they should care? Significance made explicit?
        3. Prior work surveyed and critiqued — is it clear WHY prior work is unsatisfactory?
        4. New approach introduced in the intro (not buried in methods)?
        5. Is it clear why this work is better and in what specific ways?

        ## Kajiya Test ("dynamite intro") — can any reader quickly determine:
        - What the paper is about?
        - What problem it solves?
        - Why the problem is interesting?
        - What is genuinely new?
        - Why it's exciting?

        ## Tone (Freeman/Efros)
        - Is competing work described generously, from security not competition?
        - Are novelty claims scrupulously honest?
        - Is there a "future work" section at the end? (Flag: very weak ending.)

        Output: one-line SUMMARY + SEVERITY, then per-criterion findings with quotes
        and suggested rewrites.
    """).strip()

    def run(self, ctx: Context) -> AgentResult:
        t0 = time.time()
        findings = _call(ctx.client, self.SYSTEM,
                         f"PAPER:\n\n{ctx.paper_text}")
        return AgentResult(
            agent_id=self.id, agent_name=self.name,
            summary=_first_line(findings), findings=findings,
            severity=_parse_severity(findings), references_found=[],
            elapsed=time.time() - t0,
        )


class SentenceArchitectureAgent(BaseAgent):
    id = "sentences"
    name = "Sentence Architecture"
    description = "Gopen & Swan: stress positions, topic positions, subject-verb proximity"
    priority = 1

    SYSTEM = textwrap.dedent("""
        You are a scientific editor applying Gopen & Swan's structural principles.
        Readers have predictable expectations. Violating them causes comprehension
        failures that no simplification can fix.

        ## Four Structural Principles — cite specific sentences with fixes

        1. Stress Position: important new information belongs at the END of a sentence.
           Find sentences that end flatly and bury the key finding mid-sentence.
           Use colons/semicolons to create secondary stress positions in long sentences.

        2. Topic Position: sentence openings should (a) declare whose story this is
           and (b) link backward to what was just said. Old info first, new info second.
           Flag sentences that open with unanchored new material, breaking the thread.

        3. Subject-Verb Separation: find sentences where a long interruptive phrase
           separates subject from verb, overloading working memory.

        4. Action in Verbs, Not Nouns: fix nominalizations —
           "performed an analysis of" → "analyzed"
           "provided an indication that" → "indicated"
           "there was an inhibition of X" → "X was inhibited"

        ## Structural-Conceptual Diagnosis
        Where structural problems likely reflect unclear thinking, say so.

        Output: one-line SUMMARY + SEVERITY, then the top 15 highest-impact fixes,
        quoting each sentence and providing the rewrite.
    """).strip()

    def run(self, ctx: Context) -> AgentResult:
        t0 = time.time()
        findings = _call(ctx.client, self.SYSTEM,
                         f"PAPER:\n\n{ctx.paper_text}")
        return AgentResult(
            agent_id=self.id, agent_name=self.name,
            summary=_first_line(findings), findings=findings,
            severity=_parse_severity(findings), references_found=[],
            elapsed=time.time() - t0,
        )


class VoiceAndTenseAgent(BaseAgent):
    id = "voice"
    name = "Voice & Tense"
    description = "Active voice · Past for methods/results · Present for facts"
    priority = 1

    SYSTEM = textwrap.dedent("""
        You are a scientific editor specializing in voice and tense.

        ## Active vs. Passive Voice
        Flag passive constructions where active would be stronger.
        Estimate the active/passive ratio as an overall health metric.
        Note: some passives are conventional and acceptable.

        ## Tense Conventions — flag violations
        - Past tense (correct): your specific methods, results, what you did
        - Present tense (correct): established scientific facts, what figures show,
          your model as a standing contribution, universal truths
        - Future tense: use sparingly; only for describing paper structure

        Common errors:
        - Present tense for specific experiments: "We train the model" → "We trained"
        - Past tense for established facts: "...was involved in memory" → "...is involved"
        - Tense shifts within a paragraph

        ## Sentence Energy (Strunk & White)
        Flag: "it is worth noting that", "as mentioned previously", "in this paper we",
        sentences starting "There is/are", "It is", "It was".

        Output: one-line SUMMARY + SEVERITY, then top 15 issues with quote + rewrite.
    """).strip()

    def run(self, ctx: Context) -> AgentResult:
        t0 = time.time()
        findings = _call(ctx.client, self.SYSTEM,
                         f"PAPER:\n\n{ctx.paper_text}")
        return AgentResult(
            agent_id=self.id, agent_name=self.name,
            summary=_first_line(findings), findings=findings,
            severity=_parse_severity(findings), references_found=[],
            elapsed=time.time() - t0,
        )


class ConcistnessAgent(BaseAgent):
    id = "conciseness"
    name = "Conciseness Audit"
    description = "Omit needless words · Nominalizations · Throat-clearing"
    priority = 1

    SYSTEM = textwrap.dedent("""
        You are a scientific editor applying Strunk & White's core rule:
        "Omit needless words. Vigorous writing is concise."

        Find the top 20 opportunities for compression, categorized:

        Category 1 — Wordy phrases with crisp equivalents:
        "due to the fact that" → "because"
        "in order to" → "to"
        "it is important to note that" → [delete]
        "despite the fact that" → "although"
        "at this point in time" → "now"
        "in the present study" → [usually delete]

        Category 2 — Nominalizations:
        "perform an analysis" → "analyze"
        "make a comparison" → "compare"
        "conduct an investigation" → "investigate"

        Category 3 — Redundancy:
        "end result", "final conclusion", "completely eliminate", "past history"
        Sentences that restate the previous sentence.

        Category 4 — Throat-clearing openers:
        "In this paper, we...", "The purpose of this study is to..."

        Category 5 — Hedging clutter:
        Excessive "somewhat", "rather", "quite", "in some sense"

        For each: quote original, compressed version, word savings.
        Give total estimated word reduction at the end.

        Output: one-line SUMMARY + SEVERITY, then findings.
    """).strip()

    def run(self, ctx: Context) -> AgentResult:
        t0 = time.time()
        findings = _call(ctx.client, self.SYSTEM,
                         f"PAPER:\n\n{ctx.paper_text}")
        return AgentResult(
            agent_id=self.id, agent_name=self.name,
            summary=_first_line(findings), findings=findings,
            severity=_parse_severity(findings), references_found=[],
            elapsed=time.time() - t0,
        )


class ParagraphQualityAgent(BaseAgent):
    id = "paragraphs"
    name = "Paragraph Quality"
    description = "Topic sentences · Unity · Flow · Reader-first (Knuth)"
    priority = 1

    SYSTEM = textwrap.dedent("""
        You are a writing editor evaluating paragraph-level quality in a scientific paper.

        For each paragraph check:
        - Topic sentence: does the first sentence declare the paragraph's subject?
          Can a skimmer read only topic sentences and follow the argument?
        - Unity: does every sentence serve the topic sentence? Flag strays.
        - Logical flow: does the argument build, or just list facts?
        - Reader-first (Knuth): at each moment, does each sentence answer the reader's
          implicit next question? What does the reader know so far; what do they expect?
        - Completeness: are claims supported? Flag unsupported assertions.
        - Conciseness: flag redundant sentences that repeat what was just said.

        Output format:
        Rate each section's paragraphs:
          ✅ Strong | ⚠️ Needs work (specify) | ❌ Needs rewrite (specify)

        Identify the 5 weakest paragraphs with specific actionable revision guidance.
        Note structural issues: paragraphs to split, merge, or reorder.

        Final check: if a hurried reader skims only first sentences, do they get the story?

        Output: one-line SUMMARY + SEVERITY, then findings.
    """).strip()

    def run(self, ctx: Context) -> AgentResult:
        t0 = time.time()
        findings = _call(ctx.client, self.SYSTEM,
                         f"PAPER:\n\n{ctx.paper_text}")
        return AgentResult(
            agent_id=self.id, agent_name=self.name,
            summary=_first_line(findings), findings=findings,
            severity=_parse_severity(findings), references_found=[],
            elapsed=time.time() - t0,
        )


class AcronymAgent(BaseAgent):
    id = "acronyms"
    name = "Acronym Audit"
    description = "Every acronym defined before first use · Consistency after definition"
    priority = 1

    SYSTEM = textwrap.dedent("""
        You are a meticulous scientific editor auditing acronym and abbreviation usage.

        1. List ALL acronyms/abbreviations (2+ capital letters, or abbreviated technical terms).
        2. For each, find its FIRST appearance.
        3. Verify it is spelled out at or before that first appearance.
        4. After definition, verify the short form is used consistently.

        Present a table:
        | Acronym | Full Form | First Appears | Defined? | Issue |

        Then classify:
        - Undefined acronyms: used without definition
        - Redundant definitions: defined but rarely/never used afterward
        - Double definitions: defined more than once
        - Inconsistent usage: long and short form mixed after first definition
        - Possibly standard (may not need definition for the audience)

        Special cases:
        - Acronyms in titles/headings: still need definition in text
        - Acronyms in captions: either defined in caption or already in text
        - Abstract and body often treated independently

        Output: one-line SUMMARY + SEVERITY, then the table and classified lists.
    """).strip()

    def run(self, ctx: Context) -> AgentResult:
        t0 = time.time()
        findings = _call(ctx.client, self.SYSTEM,
                         f"PAPER:\n\n{ctx.paper_text}")
        return AgentResult(
            agent_id=self.id, agent_name=self.name,
            summary=_first_line(findings), findings=findings,
            severity=_parse_severity(findings), references_found=[],
            elapsed=time.time() - t0,
        )


class FiguresTablesAgent(BaseAgent):
    id = "figures_tables"
    name = "Figures, Tables & Captions"
    description = "Coverage · Caption completeness · Figure necessity · Table formatting"
    priority = 1

    SYSTEM = textwrap.dedent("""
        You are a scientific editor auditing figures, tables, and their captions.

        ## Figure Coverage & Necessity
        - Does the paper have an appropriate number of figures for its claims?
        - Is there a clear 'story figure' or schematic that communicates the main idea
          visually? (Freeman: "most readers will skim; figures and captions tell the story")
        - Are there figures that are redundant with others or with text?
        - Are there results or methods that are described only in text but would benefit
          from a figure or table?
        - Are all figures referenced in the main text in logical order?

        ## Caption Quality (evaluate every figure and table caption)
        For each caption, check:
        - Does the caption TITLE (first sentence) state the main finding/point of the figure,
          not just describe what it shows? (A good caption title: "Model X outperforms
          baselines across all conditions." Bad: "Results of the comparison experiment.")
        - Are all panels labeled and described?
        - Are error bars, shading, and uncertainty quantification explained?
        - Are sample sizes reported in or near the caption?
        - Are statistical test results (p-values, effect sizes) explained?
        - Is the caption self-contained? Can the figure be understood without reading
          the main text?
        - Does the caption direct the reader to what to notice?
          (Freeman: "the caption should tell the reader what to notice about the figure")

        ## Table Audit
        - Are tables properly titled with a finding-oriented title?
        - Are units specified in all columns?
        - Are significance markers explained in footnotes?
        - Is the table readable (not overly wide, columns clear)?

        ## Cross-reference Audit
        - List any figures/tables mentioned in text but apparently not present.
        - List any figures/tables present but never cited in the text.

        Output: one-line SUMMARY + SEVERITY, then per-figure/table findings.
    """).strip()

    def run(self, ctx: Context) -> AgentResult:
        t0 = time.time()
        findings = _call(ctx.client, self.SYSTEM,
                         f"PAPER:\n\n{ctx.paper_text}")
        return AgentResult(
            agent_id=self.id, agent_name=self.name,
            summary=_first_line(findings), findings=findings,
            severity=_parse_severity(findings), references_found=[],
            elapsed=time.time() - t0,
        )


class ReproducibilityAgent(BaseAgent):
    id = "reproducibility"
    name = "Reproducibility Check"
    description = "Results match code/data · Methods accurately described"
    priority = 1
    needs_code = True

    SYSTEM = textwrap.dedent("""
        You are a reproducibility reviewer. Verify that the paper accurately represents
        the provided code and data.

        ## Verification Checklist

        Quantitative Results
        - Every number (accuracy, AUC, p-values, effect sizes, sample sizes, runtimes)
          should appear in or be derivable from the code/data.
        - Flag numbers that cannot be traced to a specific code path or data file.
        - Flag discrepancies between numbers in different sections.

        Methods vs. Code
        - Does the methods section accurately describe what the code does?
        - Flag: steps described in paper but absent from code.
        - Flag: steps in code not mentioned in paper.
        - Flag: parameters (learning rate, threshold, window size, etc.) described
          differently than implemented.
        - Flag: statistical tests described differently than implemented.

        Data Descriptions
        - Do sample sizes, demographics, inclusion/exclusion criteria match the data?
        - Are preprocessing steps fully described?

        Output format:
        ✅ Verified: [claim] — confirmed by [code location/data file]
        ⚠️ Unverifiable: [claim] — cannot confirm without [missing element]
        ❌ Discrepancy: paper says [X], code/data shows [Y]
        🔍 Risk: plausible but not fully checkable

        Output: one-line SUMMARY + SEVERITY, then the checklist.
    """).strip()

    def run(self, ctx: Context) -> AgentResult:
        t0 = time.time()
        if not ctx.code_text.strip():
            return AgentResult(
                agent_id=self.id, agent_name=self.name,
                summary="Skipped — no code/data provided",
                findings="No code or data was provided. Pass --code-file to enable this agent.",
                severity="ok", references_found=[], elapsed=0.0,
            )
        user = f"PAPER:\n\n{ctx.paper_text}\n\nCODE / DATA:\n\n{ctx.code_text}"
        findings = _call(ctx.client, self.SYSTEM, user)
        return AgentResult(
            agent_id=self.id, agent_name=self.name,
            summary=_first_line(findings), findings=findings,
            severity=_parse_severity(findings), references_found=[],
            elapsed=time.time() - t0,
        )


# ===========================================================================
#  PHASE 2 — whole-document agents that may use prior results
# ===========================================================================

class ConsistencyAgent(BaseAgent):
    id = "consistency"
    name = "Internal Consistency"
    description = "Terminology · Numbers · Claims across sections"
    priority = 2

    SYSTEM = textwrap.dedent("""
        You are a meticulous scientific editor checking for internal consistency.

        ## Terminology Consistency
        - Are key concepts referred to by ONE consistent name throughout?
        - Are method/model/dataset names consistent?
        - Are statistical terms used consistently?
        - Flag cases where inconsistent terminology could make readers think two things
          are different when they're the same (or vice versa).

        ## Numerical Consistency — check every number appearing in multiple places
        - Sample sizes (abstract vs. methods vs. results vs. tables)
        - Performance metrics (text vs. tables vs. figures)
        - Parameter values (methods vs. appendix)
        - Abstract numbers vs. results section numbers

        ## Claim Consistency
        - Do discussion claims match what results actually showed?
        - Does the introduction promise something the paper doesn't fully deliver?
        - Are limitations acknowledged consistently across sections?
        - Do baseline comparisons in intro match those actually reported?

        ## Figure/Table Consistency
        - Do captions accurately describe what is shown?
        - Do in-text numbers match tables/figures?

        Output format:
        ✅ Consistent: [verified claim]
        ⚠️ Discrepancy: [where / what differs — quote both]

        Output: one-line SUMMARY + SEVERITY, then findings.
    """).strip()

    def run(self, ctx: Context) -> AgentResult:
        t0 = time.time()
        findings = _call(ctx.client, self.SYSTEM,
                         f"PAPER:\n\n{ctx.paper_text}")
        return AgentResult(
            agent_id=self.id, agent_name=self.name,
            summary=_first_line(findings), findings=findings,
            severity=_parse_severity(findings), references_found=[],
            elapsed=time.time() - t0,
        )


class DiscussionAgent(BaseAgent):
    id = "discussion"
    name = "Discussion & Related Work"
    description = "Coverage of related literature · Positioning · Limitations · Future directions"
    priority = 2

    SYSTEM = textwrap.dedent("""
        You are an expert reviewer evaluating the Discussion and Related Work sections
        of a scientific paper.

        ## Positioning in the Literature
        - Does the discussion clearly articulate how this work advances the field?
        - Is the paper positioned relative to the most important prior work?
        - Are there obvious related papers or lines of work that are not discussed?
          (Flag as gaps — these are likely to be raised by reviewers.)
        - Are direct comparisons made where comparison is possible?
        - Does the paper distinguish itself from closely related work with specificity,
          not vague claims like "our method is better"?

        ## Related Work Section Quality
        - Is related work merely listed, or is it synthesized into a coherent narrative?
        - Is the framing generous and accurate (Freeman: "written from a position of
          security, not competition")?
        - Does related work section explain WHY prior methods fall short, motivating
          the new approach?
        - Are the most recent and relevant papers included?

        ## Adelson's Formula Applied to Discussion
        The discussion should mirror the introduction: you stated a problem and solution;
        now confirm what was delivered, compare to prior methods, and situate the work.

        ## Limitations
        - Are limitations acknowledged honestly and specifically?
        - Are they appropriately scoped (not underselling OR overselling)?
        - Are failure modes addressed?

        ## Scope and Impact
        - Does the discussion communicate the broader significance?
        - Does it suggest where the work may lead WITHOUT a laundry-list "future work"
          section? (Freeman: "I can't stand future work sections.")
        - Does the paper end with a strong conclusion, not a whimper?

        Output: one-line SUMMARY + SEVERITY, then section-by-section findings with
        specific suggestions and any obvious missing related work topics.
    """).strip()

    def run(self, ctx: Context) -> AgentResult:
        t0 = time.time()
        findings = _call(ctx.client, self.SYSTEM,
                         f"PAPER:\n\n{ctx.paper_text}")
        return AgentResult(
            agent_id=self.id, agent_name=self.name,
            summary=_first_line(findings), findings=findings,
            severity=_parse_severity(findings), references_found=[],
            elapsed=time.time() - t0,
        )


class MissingReferencesAgent(BaseAgent):
    """
    Paragraph-by-paragraph scan for claims that need citations.
    For each uncited claim found, queries Google Scholar via `scholarly`
    and returns candidate references with suggested insertion points.
    """
    id = "missing_refs"
    name = "Missing References"
    description = "Finds uncited claims · Searches Google Scholar for candidates"
    priority = 2
    needs_scholar = True

    IDENTIFY_SYSTEM = textwrap.dedent("""
        You are a scientific editor identifying claims that need citations.

        For the paragraph below, list every statement that:
        - Makes a factual claim about the world, prior work, or prior results
        - Quantifies something (rates, prevalence, performance benchmarks)
        - Attributes a method, concept, or finding to prior work — even implicitly
        - Describes what "has been shown", "is known", "is well-established"
        - Describes limitations of prior approaches

        EXCLUDE:
        - Claims clearly about the authors' own work presented in this paper
        - Methodological descriptions of the authors' own procedure
        - Obvious common knowledge needing no citation

        For each claim found, output EXACTLY this format (one per line):
        CLAIM: <verbatim short phrase from the paragraph>
        QUERY: <3-5 word Google Scholar search query to find the right reference>

        If no claims need references, output: NONE
    """).strip()

    def _split_paragraphs(self, text: str) -> list[tuple[int, str]]:
        """Return (paragraph_number, text) tuples, skipping references section."""
        in_refs = False
        paragraphs = []
        for i, block in enumerate(re.split(r'\n{2,}', text)):
            block = block.strip()
            if not block:
                continue
            if re.match(r'^(references|bibliography|works cited)', block, re.I):
                in_refs = True
            if in_refs:
                continue
            if len(block) > 60:   # skip short headers
                paragraphs.append((i + 1, block))
        return paragraphs

    def _scholar_search(self, query: str, n: int = 3) -> list[dict]:
        """Return up to n Scholar results as dicts with title/authors/year/url."""
        try:
            from scholarly import scholarly as sch
            results = []
            for pub in sch.search_pubs(query):
                bib = pub.get("bib", {})
                results.append({
                    "title": bib.get("title", ""),
                    "authors": bib.get("author", ""),
                    "year": bib.get("pub_year", ""),
                    "venue": bib.get("venue", ""),
                    "url": pub.get("pub_url", ""),
                    "abstract": bib.get("abstract", "")[:200],
                })
                if len(results) >= n:
                    break
            return results
        except Exception as e:
            return [{"title": f"[Scholar search failed: {e}]",
                     "authors": "", "year": "", "venue": "", "url": "", "abstract": ""}]

    def run(self, ctx: Context) -> AgentResult:
        t0 = time.time()
        paragraphs = self._split_paragraphs(ctx.paper_text)

        all_claims: list[dict] = []   # {para_num, para_text, claim, query, results}
        report_lines: list[str] = []

        for para_num, para_text in paragraphs:
            resp = _call(ctx.client, self.IDENTIFY_SYSTEM,
                         f"PARAGRAPH {para_num}:\n\n{para_text}",
                         max_tokens=512)

            if resp.strip().upper() == "NONE":
                continue

            claims_in_para: list[tuple[str, str]] = []
            for line in resp.splitlines():
                if line.startswith("CLAIM:"):
                    claim_text = line[6:].strip()
                elif line.startswith("QUERY:"):
                    query_text = line[6:].strip()
                    if claim_text:
                        claims_in_para.append((claim_text, query_text))
                        claim_text = ""
                else:
                    claim_text = ""

            if not claims_in_para:
                continue

            report_lines.append(f"\n### Paragraph {para_num}")
            report_lines.append(f"> {para_text[:200]}{'...' if len(para_text) > 200 else ''}\n")

            for claim, query in claims_in_para:
                report_lines.append(f"**Uncited claim:** {claim}")
                report_lines.append(f"*Search query:* `{query}`")
                scholar_hits = self._scholar_search(query, n=3)
                for j, hit in enumerate(scholar_hits, 1):
                    if not hit["title"].startswith("[Scholar"):
                        ref_str = (f"{j}. **{hit['title']}** — "
                                   f"{hit['authors']} ({hit['year']}) "
                                   f"*{hit['venue']}*")
                        if hit["url"]:
                            ref_str += f" [link]({hit['url']})"
                        report_lines.append(ref_str)
                        all_claims.append({
                            "para": para_num,
                            "claim": claim,
                            "ref": hit,
                        })
                    else:
                        report_lines.append(f"  *(Scholar unavailable: {hit['title']})*")
                report_lines.append("")
                time.sleep(0.5)  # be polite to Scholar

        if not report_lines:
            findings = "No obviously uncited claims found. The paper appears well-referenced."
            severity = "ok"
        else:
            n_claims = sum(1 for r in report_lines if r.startswith("**Uncited claim:**"))
            header = (f"Found **{n_claims} uncited claim(s)** across "
                      f"{len(paragraphs)} paragraphs.\n\n")
            findings = header + "\n".join(report_lines)
            severity = "major" if n_claims > 8 else "moderate" if n_claims > 3 else "minor"

        return AgentResult(
            agent_id=self.id, agent_name=self.name,
            summary=f"Found {sum(1 for r in report_lines if r.startswith('**Uncited claim'))} uncited claims",
            findings=findings,
            severity=severity,
            references_found=all_claims,
            elapsed=time.time() - t0,
        )


class ReferenceQualityAgent(BaseAgent):
    """
    Evaluates references already present in the paper:
    correctness, appropriateness, and whether better references exist.
    """
    id = "ref_quality"
    name = "Reference Quality & Correctness"
    description = "Checks cited refs for accuracy, appropriateness, and whether better refs exist"
    priority = 2
    needs_scholar = True

    EXTRACT_SYSTEM = textwrap.dedent("""
        Extract the reference list from the paper.
        For each reference output EXACTLY:
        REF_NUM: <number or key>
        TITLE: <title>
        AUTHORS: <authors>
        YEAR: <year>
        VENUE: <journal or conference>
        ---
        If no reference list is present, output: NO_REFS
    """).strip()

    ASSESS_SYSTEM = textwrap.dedent("""
        You are a senior scientific reviewer assessing whether a reference is:
        1. Correctly cited (title/authors/year match the real paper)
        2. The RIGHT reference for the claim being made (is it the seminal work?
           the most recent? the most directly relevant?)
        3. Whether a better or more canonical reference exists for this claim

        Context provided: the citing sentence in the paper, the reference details,
        and (if available) Google Scholar verification data.

        Output:
        STATUS: ok | wrong_paper | better_exists | unverifiable
        NOTES: <one sentence explanation>
        BETTER_REF: <suggest better ref title/authors if applicable>
    """).strip()

    def _extract_references(self, ctx: Context) -> list[dict]:
        resp = _call(ctx.client, self.EXTRACT_SYSTEM,
                     f"PAPER:\n\n{ctx.paper_text}", max_tokens=2048)
        if "NO_REFS" in resp:
            return []
        refs = []
        current: dict = {}
        for line in resp.splitlines():
            for field in ["REF_NUM", "TITLE", "AUTHORS", "YEAR", "VENUE"]:
                if line.startswith(f"{field}:"):
                    current[field.lower()] = line[len(field) + 1:].strip()
            if line.strip() == "---" and current:
                refs.append(current)
                current = {}
        return refs

    def _find_citing_sentence(self, ref_num: str, text: str) -> str:
        """Find the sentence(s) in the paper that cite this reference."""
        patterns = [
            rf'\[{re.escape(ref_num)}\]',
            rf'\({re.escape(ref_num)}\)',
            rf'\b{re.escape(ref_num)}\b',
        ]
        for pat in patterns:
            matches = re.findall(r'[^.!?]*' + pat + r'[^.!?]*[.!?]', text)
            if matches:
                return " | ".join(matches[:2])
        return "(citing context not found)"

    def run(self, ctx: Context) -> AgentResult:
        t0 = time.time()
        refs = self._extract_references(ctx)
        if not refs:
            return AgentResult(
                agent_id=self.id, agent_name=self.name,
                summary="No reference list found in paper",
                findings="Could not extract a reference list. Check that references are included.",
                severity="moderate", references_found=[], elapsed=time.time() - t0,
            )

        report_lines = [f"Assessed **{len(refs)} references**.\n"]
        issues = 0

        for ref in refs:
            ref_num = ref.get("ref_num", "?")
            title = ref.get("title", "")
            citing = self._find_citing_sentence(ref_num, ctx.paper_text)

            # Scholar verification
            scholar_info = ""
            if title:
                try:
                    from scholarly import scholarly as sch
                    hits = list(sch.search_pubs(title))[:1]
                    if hits:
                        bib = hits[0].get("bib", {})
                        scholar_info = (
                            f"Scholar: '{bib.get('title', '')}' "
                            f"({bib.get('pub_year', '')}) "
                            f"{bib.get('venue', '')}"
                        )
                    time.sleep(0.5)
                except Exception:
                    pass

            user_content = (
                f"CITING SENTENCE: {citing}\n\n"
                f"REFERENCE DETAILS:\n"
                f"  Title: {title}\n"
                f"  Authors: {ref.get('authors', '')}\n"
                f"  Year: {ref.get('year', '')}\n"
                f"  Venue: {ref.get('venue', '')}\n"
            )
            if scholar_info:
                user_content += f"\nSCHOLAR VERIFICATION: {scholar_info}\n"

            assessment = _call(ctx.client, self.ASSESS_SYSTEM, user_content,
                               max_tokens=200)

            status = "ok"
            for line in assessment.splitlines():
                if line.startswith("STATUS:"):
                    status = line[7:].strip()

            icon = "✅" if status == "ok" else "⚠️" if status == "unverifiable" else "❌"
            if status != "ok":
                issues += 1

            report_lines.append(f"{icon} **[{ref_num}]** {title} ({ref.get('year', '')})")
            report_lines.append(f"   {assessment.strip()}")
            report_lines.append("")

        severity = "major" if issues > 5 else "moderate" if issues > 2 else "minor" if issues > 0 else "ok"

        return AgentResult(
            agent_id=self.id, agent_name=self.name,
            summary=f"{issues} reference issue(s) found across {len(refs)} references",
            findings="\n".join(report_lines),
            severity=severity, references_found=[],
            elapsed=time.time() - t0,
        )


# ===========================================================================
#  PHASE 3 — Orchestrator / synthesis
# ===========================================================================

class OrchestratorAgent(BaseAgent):
    id = "orchestrator"
    name = "Synthesis & Prioritized Action Plan"
    description = "Synthesizes all agent findings into a ranked action plan"
    priority = 3

    SYSTEM = textwrap.dedent("""
        You are a senior scientific editor. You have received review reports from multiple
        specialized agents that have audited different dimensions of a manuscript.

        Your task:
        1. Synthesize all findings into a coherent overall assessment.
        2. Identify cross-cutting themes (e.g., "the writing problems and the structural
           problems share a common root: the main contribution is never stated clearly").
        3. Produce a PRIORITIZED ACTION PLAN — the top 10 most impactful revisions,
           ordered from most to least important.
        4. Give an overall readiness verdict:
           - Not ready: fundamental problems require major restructuring
           - Needs revision: several moderate issues; revisions achievable in one pass
           - Nearly ready: minor issues only; one careful editing pass suffices
           - Ready to submit: only cosmetic issues remain

        Format the action plan as:
        ### Action Plan
        1. [SEVERITY] Area: specific action
        ...

        Then give the overall verdict with a brief rationale (2–3 sentences).
    """).strip()

    def run(self, ctx: Context) -> AgentResult:
        t0 = time.time()
        prior_text = ""
        for agent_id, result in ctx.prior_results.items():
            if result and result.findings:
                prior_text += f"\n\n## {result.agent_name} ({result.severity.upper()})\n"
                prior_text += result.findings[:1500]   # truncate to fit context

        user = (f"PAPER EXCERPT (first 1500 chars):\n{ctx.paper_text[:1500]}\n\n"
                f"AGENT REPORTS:\n{prior_text}")
        findings = _call(ctx.client, self.SYSTEM, user, max_tokens=2048)
        return AgentResult(
            agent_id=self.id, agent_name=self.name,
            summary=_first_line(findings), findings=findings,
            severity=_parse_severity(findings), references_found=[],
            elapsed=time.time() - t0,
        )


# ===========================================================================
#  Registry
# ===========================================================================

ALL_AGENTS: list[BaseAgent] = [
    # Phase 1
    VSNCAgent(),
    IntroductionAgent(),
    SentenceArchitectureAgent(),
    VoiceAndTenseAgent(),
    ConcistnessAgent(),
    ParagraphQualityAgent(),
    AcronymAgent(),
    FiguresTablesAgent(),
    ReproducibilityAgent(),
    # Phase 2
    ConsistencyAgent(),
    DiscussionAgent(),
    MissingReferencesAgent(),
    ReferenceQualityAgent(),
    # Phase 3
    OrchestratorAgent(),
]

AGENT_REGISTRY: dict[str, BaseAgent] = {a.id: a for a in ALL_AGENTS}
