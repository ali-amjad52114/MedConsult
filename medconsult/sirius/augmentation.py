"""
SiriuS augmentation: re-run agents with evaluator feedback when score ≤ 2.
Injects issues and improvements into prompts; retries up to max_retries.
"""


class AugmentationLoop:
    """Re-runs Analyst→Clinician→Critic with Evaluator feedback when score ≤ 2."""

    def __init__(self, pipeline, max_retries: int = 2):
        self.pipeline = pipeline
        self.max_retries = max_retries

    def augment(self, original_result: dict, evaluation: dict, image=None) -> dict:
        """
        Re-run the agent chain with targeted feedback when score <= 2.
        Returns a dict with the best result found, attempt count, and score delta.
        """
        original_score = evaluation["score"]
        current_result = original_result
        current_evaluation = evaluation
        current_score = original_score
        attempts = 0

        for _ in range(self.max_retries):
            attempts += 1

            issues = current_evaluation.get("issues", [])
            improvements = current_evaluation.get("improvements", [])

            analyst_feedback = self._build_feedback(issues, improvements)
            clinician_feedback = self._build_feedback(issues, improvements)
            critic_feedback = self._build_feedback(issues, improvements)

            user_input = original_result["input"]

            # Re-run chain with feedback injected into each agent
            analyst_out = self.pipeline.analyst.analyze(
                user_input, image, feedback=analyst_feedback
            )
            clinician_out = self.pipeline.clinician.interpret(
                user_input, analyst_out, image, feedback=clinician_feedback
            )
            critic_out = self.pipeline.critic.review_and_communicate(
                user_input, analyst_out, clinician_out, image, feedback=critic_feedback
            )

            new_result = {
                "input": user_input,
                "analyst": analyst_out,
                "clinician": clinician_out,
                "critic": critic_out,
                "metadata": original_result.get("metadata", {}),
            }

            new_evaluation = self.pipeline.evaluator.evaluate(
                user_input, analyst_out, clinician_out, critic_out
            )
            new_score = new_evaluation["score"]

            # Accept if improved significantly or threshold reached
            if new_score >= 4 or new_score > current_score:
                return {
                    "improved_result": new_result,
                    "improved_evaluation": new_evaluation,
                    "attempts": attempts,
                    "original_score": original_score,
                    "final_score": new_score,
                    "improved": new_score > original_score,
                }

            current_result = new_result
            current_evaluation = new_evaluation
            current_score = new_score

        return {
            "improved_result": current_result,
            "improved_evaluation": current_evaluation,
            "attempts": attempts,
            "original_score": original_score,
            "final_score": current_score,
            "improved": current_score > original_score,
        }

    def _build_feedback(self, issues: list, improvements: list) -> str | None:
        parts = []
        if issues:
            parts.append("Issues to fix: " + "; ".join(issues[:3]))
        if improvements:
            parts.append("Improvements needed: " + "; ".join(improvements[:3]))
        return " | ".join(parts) if parts else None
