"""Escalating augmentation strategies for failed chains."""

def feedback_injection(feedback):
    """Strategy 1: Inject evaluator feedback into prompt."""
    return (
        f"\n⚠️ IMPROVEMENT FEEDBACK: {feedback}\n"
        "Please address this feedback in your response.\n"
    )

def verbose_mode(feedback):
    """Strategy 2: Switch to detailed/verbose prompt."""
    return (
        f"\n⚠️ SWITCHING TO DETAILED MODE.\n"
        f"FEEDBACK: {feedback}\n"
        "REQUIREMENTS:\n"
        "1. Be EXTREMELY thorough — list every single finding\n"
        "2. Double-check every reference range\n"
        "3. Explicitly state reasoning for each conclusion\n"
        "4. If uncertain, say so explicitly\n"
    )

def decomposition(feedback):
    """Strategy 3: Chain-of-thought sub-task decomposition."""
    return (
        f"\n⚠️ STEP-BY-STEP DECOMPOSITION MODE.\n"
        f"FEEDBACK: {feedback}\n"
        "STEP 1: List ALL values found in the input\n"
        "STEP 2: For each value, state reference range and normal/high/low\n"
        "STEP 3: Identify PATTERNS across multiple values\n"
        "STEP 4: Generate differential diagnoses based on patterns\n"
        "STEP 5: Assess urgency level\n"
        "STEP 6: Write your final analysis\n"
    )

STRATEGIES = [feedback_injection, verbose_mode, decomposition]

def get_strategy(attempt):
    """Return the strategy function for the given attempt number (1-indexed)."""
    idx = min(attempt - 1, len(STRATEGIES) - 1)
    return STRATEGIES[idx]
