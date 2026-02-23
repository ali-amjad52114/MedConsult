"""
Distills successful analysis chains into structured medical lessons using Gemini.
Lessons capture reasoning patterns for persistent in-context learning.
"""

import re

LESSON_EXTRACTION_PROMPT = """You extract structured medical reasoning lessons from successful AI analysis chains. Each lesson captures a REASONING PATTERN that should be reused in future analyses.

Given a complete analysis chain, extract 2-5 lessons.

FORMAT EACH LESSON EXACTLY AS:

<lesson>
  <topic>[specific medical topic, e.g. "CBC anemia triad recognition"]</topic>
  <input_type>[lab_report | clinical_note | imaging | general]</input_type>
  <rule>[the reasoning pattern — what to look for, how to interpret it, what to do. Be specific and actionable. Max 100 words.]</rule>
  <example_values>[specific values from this chain that triggered the rule]</example_values>
  <confidence>[high | medium]</confidence>
</lesson>

FOCUS ON:
- Pattern recognition rules (e.g., "when X and Y are both abnormal, consider Z")
- Common pitfalls to avoid (e.g., "don't alarm patient about isolated mild elevation")
- Communication lessons (e.g., "always explain this term in plain language")
- Clinical reasoning chains (e.g., "check MCV to sub-classify anemia type")

DO NOT extract obvious facts. Extract REASONING PATTERNS that improve future analyses."""


class LessonExtractor:
    """Uses Gemini to distill 2–5 reasoning lessons from a successful chain."""

    def __init__(self, cloud_manager):
        self.cloud_manager = cloud_manager

    def extract(self, chain_data: dict) -> list:
        """
        Sends full chain (input, analyst, clinician, critic, score) to Gemini.
        Returns a list of lesson dicts. Returns [] on failure.
        """
        user_message = (
            f"ORIGINAL INPUT:\n{chain_data.get('input', '')}\n\n"
            f"ANALYST OUTPUT:\n{chain_data.get('analyst', '')}\n\n"
            f"CLINICIAN OUTPUT:\n{chain_data.get('clinician', '')}\n\n"
            f"CRITIC OUTPUT:\n{chain_data.get('critic', '')}\n\n"
            f"EVALUATION SCORE: {chain_data.get('evaluation', {}).get('score', 'N/A')}\n\n"
            f"Extract 2-5 reasoning lessons from this chain."
        )

        try:
            response = self.cloud_manager.generate_response(
                system_prompt=LESSON_EXTRACTION_PROMPT,
                user_message=user_message,
                max_tokens=2048,
            )
        except Exception as e:
            print(f"Warning: Lesson extraction failed: {e}")
            return []

        return self._parse_lessons(response, chain_data)

    def _parse_lessons(self, response: str, chain_data: dict) -> list:
        lessons = []
        lesson_blocks = re.findall(r"<lesson>(.*?)</lesson>", response, re.DOTALL)

        source_score = chain_data.get("evaluation", {}).get("score", 3)
        chain_id = chain_data.get("metadata", {}).get("timestamp", "unknown")

        for block in lesson_blocks:
            try:
                rule = self._extract_tag(block, "rule")
                if not rule:
                    continue
                lessons.append({
                    "topic": self._extract_tag(block, "topic"),
                    "input_type": self._extract_tag(block, "input_type"),
                    "rule": rule,
                    "example_values": self._extract_tag(block, "example_values"),
                    "confidence": self._extract_tag(block, "confidence") or "medium",
                    "source_score": source_score,
                    "chain_id": chain_id,
                })
            except Exception:
                continue

        return lessons

    def _extract_tag(self, text: str, tag: str) -> str:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else ""
