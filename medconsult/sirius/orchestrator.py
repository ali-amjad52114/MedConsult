"""
SiriuS orchestration: evaluate, extract lessons, store, and return display data.
"""

from sirius.lesson_extractor import extract_lessons_from_evaluation
from sirius.memory_store import add_lessons, get_count, get_all_topics
from sirius.memory_retriever import get_relevant_lessons, format_memory_context
from sirius.augmentation import should_trigger_augmentation


def evaluate_and_learn(
    cloud_manager,
    original_input: str,
    analyst_output: str,
    clinician_output: str,
    critic_output: str,
    lessons_injected: list[dict],
) -> dict | None:
    """
    Evaluates the pipeline output, extracts lessons, stores them.
    Returns display data or None on failure (never raises).
    """
    try:
        from agents.evaluator import EvaluatorAgent

        evaluator = EvaluatorAgent(cloud_manager)
        result = evaluator.evaluate(
            original_input, analyst_output, clinician_output, critic_output
        )
        score = result.get("score", 3)
        raw_evaluation = result.get("raw_evaluation", "")
        augmentation_triggered = should_trigger_augmentation(score)

        # Extract and store lessons
        topics_extracted = extract_lessons_from_evaluation(raw_evaluation)
        add_lessons(topics_extracted)
        total_lessons = get_count()

        lessons_injected_topics = [
            l.get("topic", "") for l in lessons_injected if l.get("topic")
        ]

        return {
            "score": score,
            "raw_evaluation": raw_evaluation,
            "augmentation_triggered": augmentation_triggered,
            "lessons_extracted": topics_extracted,
            "lessons_injected": lessons_injected_topics,
            "total_lessons": total_lessons,
            "ok": True,
        }
    except Exception as e:
        print(f"SiriuS evaluate_and_learn failed: {e}")
        return None


def retrieve_memory_context(original_input: str, limit: int = 5) -> tuple[str, list[dict]]:
    """
    Retrieves relevant lessons for injection.
    Returns (formatted_context, list_of_lesson_dicts).
    """
    lessons = get_relevant_lessons(original_input, limit=limit)
    return format_memory_context(lessons), lessons
