"""LLM-as-judge: a separate Groq/Anthropic call that critiques the briefing.

The judge sees the briefing, the source context, and a strict rubric. It
returns per-dimension 0-1 scores with one-sentence critiques in XML.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from financial_agent.evaluation.rubric import DIMENSIONS
from financial_agent.reasoning.briefing import Briefing
from financial_agent.reasoning.context import ReasoningContext


JUDGE_SYSTEM_PROMPT = """You are a strict investment-committee reviewer evaluating a portfolio briefing.

Score the briefing across these five dimensions on a 0.0 - 1.0 scale:

1. **Causal Depth** — Does each <link> traverse macro → sector → stock → portfolio with
   non-empty fields, or are steps skipped or hand-waved?
2. **Evidence Accuracy** — Are all cited news IDs present in the provided context?
   Do quoted ₹ figures match the contribution_attribution data?
3. **Conflict Handling** — Are the divergences and conflicting news in the context
   addressed by name in <conflict> blocks, with a real reconciliation?
4. **Prioritization** — Does the chain focus on the largest ₹ contributors, or scatter
   across small effects?
5. **Quantification** — Does every link cite an exact ₹ figure and a % weight, or
   does it use vague language ("significant drop", "weighed down")?

Be strict. A 0.7 means "competent but generic." Reserve 0.9+ for briefings that
genuinely teach the reader something. Penalize hallucinated news IDs or made-up
percentages aggressively (≤ 0.3 on Evidence Accuracy).

Respond with EXACTLY this XML structure (no markdown, no preamble):

<evaluation>
  <dimension name="Causal Depth" score="0.X">One-sentence critique.</dimension>
  <dimension name="Evidence Accuracy" score="0.X">One-sentence critique.</dimension>
  <dimension name="Conflict Handling" score="0.X">One-sentence critique.</dimension>
  <dimension name="Prioritization" score="0.X">One-sentence critique.</dimension>
  <dimension name="Quantification" score="0.X">One-sentence critique.</dimension>
  <summary>One sentence overall verdict.</summary>
</evaluation>
"""


JUDGE_USER_TEMPLATE = """Evaluate this briefing.

# Briefing
```xml
{briefing_xml}
```

# Source context (what the agent had access to)
```json
{context_json}
```

Output the <evaluation> XML only.
"""


@dataclass(frozen=True)
class JudgeScore:
    name: str
    score: float
    critique: str


@dataclass(frozen=True)
class JudgeOutput:
    scores: dict[str, JudgeScore]   # keyed by dimension name
    summary: str
    raw_response: str


def render_judge_prompt(briefing: Briefing, context: ReasoningContext) -> str:
    briefing_xml = briefing.raw_response or "(unavailable)"
    context_json = json.dumps(context.to_dict(), indent=2, ensure_ascii=False)
    return JUDGE_USER_TEMPLATE.format(
        briefing_xml=briefing_xml,
        context_json=context_json,
    )


# ---------- Parsing ---------------------------------------------------------

class JudgeParseError(ValueError):
    """Raised when the judge's XML response cannot be coerced into scores."""


def _extract_evaluation_xml(raw: str) -> str:
    cleaned = re.sub(r"```(?:xml)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    match = re.search(r"<evaluation>.*?</evaluation>", cleaned, flags=re.DOTALL)
    if not match:
        raise JudgeParseError("No <evaluation>...</evaluation> block in judge output.")
    return match.group(0)


def parse_judge_output(raw: str) -> JudgeOutput:
    xml_text = _extract_evaluation_xml(raw)
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise JudgeParseError(f"Malformed judge XML: {exc}") from exc

    scores: dict[str, JudgeScore] = {}
    for elem in root.findall("dimension"):
        name = elem.attrib.get("name", "").strip()
        if name not in DIMENSIONS:
            continue
        try:
            score = float(elem.attrib.get("score", "0"))
        except ValueError:
            score = 0.0
        score = max(0.0, min(1.0, score))
        critique = (elem.text or "").strip()
        scores[name] = JudgeScore(name=name, score=score, critique=critique)

    summary_elem = root.find("summary")
    summary = (summary_elem.text or "").strip() if summary_elem is not None else ""

    return JudgeOutput(scores=scores, summary=summary, raw_response=raw)
