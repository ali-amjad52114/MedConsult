"""
The Critic Agent in the MedConsult pipeline.
Reviews the Clinician's interpretation and produces a patient-friendly report.
"""

from prompts.critic_prompt import CRITIC_SYSTEM_PROMPT


class CriticAgent:
    """Reviews clinician output and produces plain-language patient summary."""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def review_and_communicate(
        self,
        original_input: str,
        analyst_output: str,
        clinician_output: str,
        image=None,
        feedback=None,
        memory_context=None,
    ) -> str:
        """
        Reviews clinician output and produces a plain-language patient report.
        feedback and memory_context are optional, used by later SiriuS phases.
        """
        # Prepend memory_context and feedback same as Analyst/Clinician
        enhanced = ""
        if memory_context:
            enhanced += memory_context + "\n\n"
        if feedback:
            enhanced += f"⚠️ QUALITY FEEDBACK: {feedback}\n\n"

        # Full chain: original + Analyst + Clinician → Critic produces patient summary
        enhanced += f"""ORIGINAL MEDICAL INPUT:
{original_input}

ANALYST EXTRACTION:
{analyst_output}

CLINICIAN INTERPRETATION:
{clinician_output}

Based on all of the above, provide your critical review and patient-friendly summary."""

        return self.model_manager.generate_response(
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_message=enhanced,
            image=image,
            max_tokens=2048,
        )
