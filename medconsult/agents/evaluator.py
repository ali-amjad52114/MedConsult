"""
Evaluator Agent: scores analysis chains 1–5 using Gemini cloud.
Part of SiriuS self-improvement (runs async after user gets results).
"""

import re
from prompts.evaluator_prompt import EVALUATOR_SYSTEM_PROMPT


class EvaluatorAgent:
    """Scores Analyst→Clinician→Critic chains 1–5 for SiriuS quality feedback."""

    def __init__(self, cloud_manager):
        self.cloud_manager = cloud_manager

    def evaluate(self, original_input, analyst_output, clinician_output, critic_output):
        """Sends full chain to Gemini; parses Score, Key Issues, Suggested Improvements."""
        prompt = f"""
Original Input:
{original_input}

Analyst Output:
{analyst_output}

Clinician Output:
{clinician_output}

Critic Output:
{critic_output}
"""
        try:
            response = self.cloud_manager.generate_response(
                system_prompt=EVALUATOR_SYSTEM_PROMPT,
                user_message=prompt.strip()
            )
        except Exception as e:
            print(f"Error during Evaluator generation: {e}")
            response = ""

        score = 3
        issues = []
        improvements = []

        if response:
            # Parse score (1–5)
            score_match = re.search(r'\*?\*?Score:\*?\*?\s*(\d)(?:/5)?', response, re.IGNORECASE)
            if score_match:
                try:
                    score = int(score_match.group(1))
                    if score < 1: score = 1
                    if score > 5: score = 5
                except ValueError:
                    pass

            # Parse Key Issues Found
            issues_match = re.search(r'Key Issues Found:\s*(.*?)(?=Suggested Improvements:|---|$)', response, re.IGNORECASE | re.DOTALL)
            if issues_match:
                issues_text = issues_match.group(1).strip()
                if issues_text.lower() != 'none':
                    issues = [line.strip('-* ').strip() for line in issues_text.split('\n') if line.strip('-* ')]

            # Parse Suggested Improvements
            improvements_match = re.search(r'Suggested Improvements:\s*(.*?)(?=---|$)', response, re.IGNORECASE | re.DOTALL)
            if improvements_match:
                improvements_text = improvements_match.group(1).strip()
                if improvements_text.lower() != 'none':
                    improvements = [line.strip('-* ').strip() for line in improvements_text.split('\n') if line.strip('-* ')]

        return {
            "score": score,
            "raw_evaluation": response.strip(),
            "issues": issues,
            "improvements": improvements
        }
