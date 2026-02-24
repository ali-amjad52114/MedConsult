"""
The Clinician Agent in the MedConsult pipeline.
Interprets medical patterns from Analyst extraction using MedGemma.
"""

from prompts.clinician_prompt import CLINICIAN_SYSTEM_PROMPT


class ClinicianAgent:
    """Interprets clinical significance from extracted medical facts."""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def interpret(
        self,
        original_input: str,
        analyst_output: str,
        image=None,
        feedback=None,
        memory_context=None,
    ) -> str:
        """
        Interprets medical significance from the analyst's extraction.
        feedback and memory_context are optional, used by later SiriuS phases.
        """
        # Prepend memory_context (lessons) and feedback (from Evaluator) if present
        enhanced = ""
        if memory_context:
            enhanced += memory_context + "\n\n"
        if feedback:
            enhanced += f"⚠️ QUALITY FEEDBACK: {feedback}\n\n"

        # Combine original input + Analyst extraction for interpretation
        enhanced += f"""ORIGINAL MEDICAL INPUT:
{original_input}

FACTUAL EXTRACTION (from Analyst):
{analyst_output}

Based on the extraction above, provide your clinical interpretation."""

        return self.model_manager.generate_response(
            system_prompt=CLINICIAN_SYSTEM_PROMPT,
            user_message=enhanced,
            image=image,
            max_tokens=2048,
        )
