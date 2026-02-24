"""Periodic validation — runs every N cases to assess quality trends."""

import json
import re
from datetime import datetime


VALIDATION_PROMPT = """You are a medical AI quality auditor. Review these {n} recent
medical analysis chains with a STRICT rubric.

For EACH chain, score 4 dimensions (1-5):
  accuracy      — values correctly extracted and interpreted?
  reasoning     — clinical patterns identified with good differentials?
  communication — patient summary clear, accurate, actionable?
  safety        — urgent/critical findings appropriately flagged?

Then give OVERALL assessment:
  trend           — "improving", "stable", or "declining"
  weakest_agent   — "analyst", "clinician", or "critic"
  strongest_agent — "analyst", "clinician", or "critic"
  key_concern     — one sentence: most important issue
  key_strength    — one sentence: what is working well

Return ONLY JSON (no markdown fences):
{{
  "chain_scores": [
    {{"id": 1, "accuracy": 4, "reasoning": 3, "communication": 5, "safety": 4}}
  ],
  "trend": "stable",
  "weakest_agent": "critic",
  "strongest_agent": "clinician",
  "key_concern": "...",
  "key_strength": "..."
}}

CHAINS:
{chains}
"""


class PeriodicValidator:
    """Validates batches of recent chains to track quality trends."""

    def __init__(self, cloud_manager, validation_interval=5):
        self.cloud = cloud_manager
        self.interval = validation_interval
        self.reports = []

    def should_validate(self, run_count):
        """Returns True if run_count is a multiple of the interval."""
        return run_count > 0 and run_count % self.interval == 0

    def run(self, chain_history):
        """Validate a list of recent chain dicts.

        Each chain dict should have keys:
          input, analyst, clinician, critic, score, name (optional)
        """
        n = len(chain_history)
        chains_text = ""
        for i, ch in enumerate(chain_history):
            chains_text += f"\n--- Chain {i+1}"
            if ch.get("name"):
                chains_text += f" ({ch['name']})"
            chains_text += " ---\n"
            chains_text += f"Input:     {str(ch.get('input', ''))[:200]}\n"
            chains_text += f"Analyst:   {str(ch.get('analyst', ''))[:200]}\n"
            chains_text += f"Clinician: {str(ch.get('clinician', ''))[:200]}\n"
            chains_text += f"Critic:    {str(ch.get('critic', ''))[:200]}\n"
            chains_text += f"Score:     {ch.get('score', '?')}/5\n"

        prompt = VALIDATION_PROMPT.format(n=n, chains=chains_text)

        try:
            raw = self.cloud.generate_response(
                system_prompt="You are a strict medical AI auditor.",
                user_message=prompt,
                max_tokens=2048
            )
            jm = re.search(r"\{[\s\S]*\}", raw)
            if jm:
                report = json.loads(jm.group())
            else:
                report = {"error": "Could not parse response", "raw": raw[:500]}
        except Exception as e:
            report = {"error": str(e)}

        report["timestamp"] = datetime.now().isoformat()
        report["num_chains"] = n
        self.reports.append(report)
        return report

    def get_latest_report(self):
        """Return the most recent validation report, or None."""
        return self.reports[-1] if self.reports else None

    def get_all_reports(self):
        """Return all validation reports for trend visualization."""
        return list(self.reports)
