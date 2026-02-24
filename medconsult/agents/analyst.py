"""
The Analyst Agent in the MedConsult pipeline.
Responsible for data extraction and normalization using MedGemma.
"""

from prompts.analyst_prompt import ANALYST_SYSTEM_PROMPT


class AnalystAgent:
    """Extracts and organizes factual findings from medical inputs."""

    def __init__(self, model_manager):
        """Stores the MedGemma model manager."""
        self.model_manager = model_manager

    def analyze(
        self,
        user_input_text: str,
        image=None,
        feedback=None,
        memory_context=None,
    ) -> str:
        """
        Extracts structured facts from raw input. NO interpretation.
        memory_context = lessons from ChromaDB (injected at top of prompt).
        feedback = evaluator issues/improvements (used by AugmentationLoop when score ≤ 2).
        """
        # Build prompt: [memory_context] + [feedback] + user_input_text
        enhanced = ""
        if memory_context:
            enhanced += memory_context + "\n\n"
        if feedback:
            enhanced += f"⚠️ QUALITY FEEDBACK: {feedback}\n\n"
        enhanced += user_input_text

        # Image ABCDE extraction needs fewer tokens than a full lab report.
        # Cap at 512 when image is present to keep inference fast on GPU.
        max_tokens = 512 if image is not None else 2048

        return self.model_manager.generate_response(
            ANALYST_SYSTEM_PROMPT,
            enhanced,
            image=image,
            max_tokens=max_tokens,
        )
