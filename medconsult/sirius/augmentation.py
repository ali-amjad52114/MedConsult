"""Enhanced augmentation with per-agent scoring, escalation,
cross-agent feedback, and anti-lesson extraction."""

import json
import re
import xml.etree.ElementTree as ET

from sirius.augmentation_strategies import get_strategy


PER_AGENT_EVAL_PROMPT = """Score EACH agent in this medical analysis (1-5).
Determine which agents should retry.

Input: {input_text}
ANALYST:   {analyst}
CLINICIAN: {clinician}
CRITIC:    {critic}

Return ONLY JSON (no markdown fences):
{{
  "analyst":   {{"score": 4, "feedback": "...", "should_retry": false}},
  "clinician": {{"score": 2, "feedback": "...", "should_retry": true}},
  "critic":    {{"score": 3, "feedback": "...", "should_retry": false}}
}}
"""

CROSS_FEEDBACK_PROMPT = """The Critic reviewed a medical analysis and found problems.
Write ONE specific sentence of feedback for the Analyst.
Start with "The Critic noted: ..."

Critic output: {critic_output}
"""

ANTI_LESSON_PROMPT = """This medical analysis scored {score}/5. Analyze what went WRONG.
Extract "anti-lessons" — specific patterns to AVOID.

Input:     {input_text}
Analyst:   {analyst}
Clinician: {clinician}
Critic:    {critic}

Return ONLY XML:
<anti_lessons>
  <lesson>
    <target_agent>clinician</target_agent>
    <topic>Missed critical value</topic>
    <rule>AVOID: Do not overlook potassium above 5.5 — always flag as critical</rule>
  </lesson>
</anti_lessons>
"""

VALID_AGENTS = {"analyst", "clinician", "critic"}


class EnhancedAugmentation:
    """Augmentation engine with 4 upgrade paths."""

    def __init__(self, cloud_manager, max_retries=3):
        self.cloud = cloud_manager
        self.max_retries = max_retries
        self.retry_log = []

    # ── (a) Per-agent scoring ──

    def score_agents(self, chain):
        """Score each agent independently. Returns dict or None."""
        prompt = PER_AGENT_EVAL_PROMPT.format(
            input_text=str(chain["input"])[:300],
            analyst=str(chain["analyst"])[:300],
            clinician=str(chain["clinician"])[:300],
            critic=str(chain["critic"])[:300],
        )
        try:
            raw = self.cloud.generate_response(
                system_prompt="You are a strict evaluator scoring medical agents.",
                user_message=prompt,
                max_tokens=1024
            )
            jm = re.search(r"\{[\s\S]*\}", raw)
            return json.loads(jm.group()) if jm else None
        except Exception:
            return None

    def get_retry_agents(self, agent_scores):
        """Return list of (agent_name, feedback) for agents needing retry."""
        retries = []
        if not agent_scores:
            return retries
        for name in ["analyst", "clinician", "critic"]:
            fb = agent_scores.get(name, {})
            if isinstance(fb, dict) and fb.get("should_retry", False):
                retries.append((name, fb.get("feedback", "")))
        return retries

    # ── (b) Escalating retry ──

    def get_retry_context(self, agent_name, feedback, attempt):
        """Return (context_string, strategy_name) for the given attempt."""
        strategy_fn = get_strategy(attempt)
        context = strategy_fn(feedback)
        strategy_name = strategy_fn.__name__
        self.retry_log.append({
            "agent": agent_name,
            "attempt": attempt,
            "strategy": strategy_name,
        })
        return context, strategy_name

    # ── (c) Cross-agent feedback ──

    def get_cross_feedback(self, critic_output):
        """Generate feedback from Critic to Analyst."""
        prompt = CROSS_FEEDBACK_PROMPT.format(
            critic_output=str(critic_output)[:400]
        )
        try:
            return self.cloud.generate_response(
                system_prompt="You pass critical feedback from a critic to an analyst.",
                user_message=prompt,
                max_tokens=200
            ).strip()
        except Exception:
            return ""

    # ── (d) Anti-lessons ──

    def extract_anti_lessons(self, chain, score):
        """Extract pitfall-warning lessons from failed chains."""
        prompt = ANTI_LESSON_PROMPT.format(
            score=score,
            input_text=str(chain["input"])[:300],
            analyst=str(chain["analyst"])[:300],
            clinician=str(chain["clinician"])[:300],
            critic=str(chain["critic"])[:300],
        )
        try:
            raw = self.cloud.generate_response(
                system_prompt="You are extracting failure patterns to avoid.",
                user_message=prompt,
                max_tokens=1024
            )
            match = re.search(
                r"<anti_lessons>(.*?)</anti_lessons>", raw, re.DOTALL | re.IGNORECASE
            )
            if not match:
                return []
            root = ET.fromstring(
                f"<anti_lessons>{match.group(1)}</anti_lessons>"
            )
            results = []
            for el in root.findall("lesson"):
                agent = (el.findtext("target_agent") or "").strip().lower()
                rule = (el.findtext("rule") or "").strip()
                if not rule:
                    continue
                results.append({
                    "target_agent": agent if agent in VALID_AGENTS else "clinician",
                    "lesson_type": "pitfall_warning",
                    "topic": (el.findtext("topic") or "failure pattern").strip(),
                    "rule": rule,
                    "confidence": "high",
                })
            return results
        except Exception:
            return []

    def get_retry_log(self):
        return list(self.retry_log)
