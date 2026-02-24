"""
Distills successful analysis chains into structured medical lessons using Gemini.
Lessons capture reasoning patterns for persistent in-context learning.
"""

import re

LESSON_EXTRACTION_PROMPT = """Analyze this medical reasoning chain and extract lessons.
For EACH lesson provide ALL fields:

1. target_agent — who benefits most:
   "analyst"   = extracting values, flagging ranges, parsing formats
   "clinician" = pattern recognition, differentials, cross-value logic
   "critic"    = patient communication, summary clarity, safety flags

2. lesson_type:
   "extraction_pattern" = how to correctly extract/flag lab values
   "reasoning_chain"    = clinical reasoning steps, differential strategies
   "communication_tip"  = explaining findings to patients clearly
   "pitfall_warning"    = common mistakes to AVOID

3. topic — brief label (3-5 words)
4. rule  — the lesson (1-2 sentences, specific, actionable)
5. confidence — "high", "medium", or "low"

Return ONLY XML:
<lessons>
  <lesson>
    <target_agent>analyst</target_agent>
    <lesson_type>extraction_pattern</lesson_type>
    <topic>Anemia triad detection</topic>
    <rule>When RBC, Hemoglobin, and Hematocrit are ALL low, flag as anemia triad</rule>
    <confidence>high</confidence>
  </lesson>
</lessons>
"""


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
            f"CHAIN DATA:\n"
            f"Input: {chain_data.get('input', '')}\n"
            f"Analyst: {chain_data.get('analyst', '')}\n"
            f"Clinician: {chain_data.get('clinician', '')}\n"
            f"Critic: {chain_data.get('critic', '')}\n"
            f"Score: {chain_data.get('evaluation', {}).get('score', 'N/A')}/5\n"
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

        return self._parse_lessons(response)

    def _parse_lessons(self, response: str) -> list:
        lessons = []
        lesson_blocks = re.findall(r"<lesson>(.*?)</lesson>", response, re.DOTALL)
        
        valid_agents = {"analyst", "clinician", "critic"}
        valid_types = {"extraction_pattern", "reasoning_chain", "communication_tip", "pitfall_warning"}

        for block in lesson_blocks:
            try:
                rule = self._extract_tag(block, "rule")
                if not rule:
                    continue
                
                target_agent = self._extract_tag(block, "target_agent").lower()
                if target_agent not in valid_agents:
                    target_agent = "clinician"

                lesson_type = self._extract_tag(block, "lesson_type").lower()
                if lesson_type not in valid_types:
                    lesson_type = "reasoning_chain"

                lessons.append({
                    "target_agent": target_agent,
                    "lesson_type": lesson_type,
                    "topic": self._extract_tag(block, "topic"),
                    "rule": rule,
                    "confidence": self._extract_tag(block, "confidence").lower() or "medium",
                })
            except Exception:
                continue

        return lessons

    def _extract_tag(self, text: str, tag: str) -> str:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""
