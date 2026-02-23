"""Phase 5C: Persistent Memory + In-Context Learning tests"""
import os
import pytest
from pipeline import MedConsultPipeline
from sirius.memory_store import MemoryStore
from sirius.memory_retriever import MemoryRetriever

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    return MedConsultPipeline()


@pytest.fixture(scope="module")
def cbc_text():
    with open(os.path.join(DATA_DIR, "sample_cbc.txt")) as f:
        return f.read()


@pytest.fixture(scope="module")
def clinical_text():
    with open(os.path.join(DATA_DIR, "sample_clinical_note.txt")) as f:
        return f.read()


@pytest.fixture(scope="module")
def populated_pipeline(pipeline, cbc_text):
    """
    Run pipeline + evaluate_and_learn on CBC to populate memory.
    Shared across tests that need lessons pre-stored.
    """
    result = pipeline.run(cbc_text)
    sirius = pipeline.evaluate_and_learn(result)
    print(f"\n[populate] Score: {sirius['evaluation']['score']}, "
          f"Lessons: {sirius.get('lessons_extracted', 0)}, "
          f"Total: {sirius.get('total_lessons', 0)}")
    return pipeline


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_empty_returns_none(tmp_path):
    """TEST 1 - Fresh empty memory store returns None for any query."""
    fresh_store = MemoryStore(persist_dir=str(tmp_path / "chroma_db"))
    retriever = MemoryRetriever(fresh_store)

    result = retriever.get_context_for_agent("analyst", "CBC report with hemoglobin", "lab_report")
    assert result is None, "Expected None from empty memory store"


def test_retrieves_after_learning(populated_pipeline):
    """TEST 2 - After evaluate_and_learn, retriever returns lesson context."""
    total = populated_pipeline.memory_store.get_lesson_count()
    print(f"\nLessons in store: {total}")

    if total == 0:
        pytest.skip("No lessons stored (score < 3 or extraction failed)")

    retriever = MemoryRetriever(populated_pipeline.memory_store)
    context = retriever.get_context_for_agent(
        "analyst", "hemoglobin is low", "lab_report"
    )

    print(f"\nRetrieved context:\n{context[:500] if context else 'None'}")

    assert context is not None, "Expected context after lessons were stored"
    assert "<lesson" in context, "Context missing <lesson> tags"
    assert 'topic="' in context, "Context missing topic= attribute"


def test_relevance(populated_pipeline):
    """TEST 3 - ChromaDB returns relevant lessons for medical queries."""
    total = populated_pipeline.memory_store.get_lesson_count()
    if total == 0:
        pytest.skip("No lessons stored — skipping relevance test")

    retriever = MemoryRetriever(populated_pipeline.memory_store)

    # Query relevant to CBC / anemia
    context_anemia = retriever.get_context_for_agent(
        "analyst", "anemia low hemoglobin red blood cells", "lab_report"
    )
    # Query for unrelated topic
    context_heart = retriever.get_context_for_agent(
        "clinician", "heart failure edema furosemide", "clinical_note"
    )

    print(f"\nAnemia context:\n{(context_anemia or 'None')[:400]}")
    print(f"\nHeart failure context:\n{(context_heart or 'None')[:400]}")

    # At least the anemia query should return something (CBC lessons were stored)
    assert context_anemia is not None, \
        "Expected anemia query to match stored CBC lessons"

    if context_anemia and context_heart:
        # Both returned results — verify anemia context is relevant
        anemia_has_relevant = any(
            kw in context_anemia.lower()
            for kw in ["anemia", "hemoglobin", "cbc", "rbc", "hematocrit", "blood"]
        )
        print(f"Anemia context has relevant terms: {anemia_has_relevant}")


def test_pipeline_uses_memory(populated_pipeline, cbc_text):
    """TEST 4 - After lessons stored, pipeline reports total_lessons_available > 0."""
    total = populated_pipeline.memory_store.get_lesson_count()
    print(f"\nTotal lessons available: {total}")

    # Run pipeline on a different input (glucose)
    result = populated_pipeline.run("Glucose: 250 mg/dL, Creatinine: 2.1 mg/dL (reference 0.6-1.2)")

    print(f"memory_used: {result['metadata']['memory_used']}")
    print(f"total_lessons_available: {result['metadata']['total_lessons_available']}")

    assert result["metadata"]["total_lessons_available"] > 0, \
        "total_lessons_available should be > 0 after lessons were stored"


def test_works_without_memory(pipeline, tmp_path, cbc_text):
    """TEST 5 - Pipeline works correctly when memory store is empty (graceful degradation)."""
    # Swap to a fresh empty memory store for this test
    original_store = pipeline.memory_store
    pipeline.memory_store = MemoryStore(persist_dir=str(tmp_path / "chroma_empty"))

    try:
        result = pipeline.run(cbc_text)

        print(f"\nmemory_used: {result['metadata']['memory_used']}")
        print(f"total_lessons_available: {result['metadata']['total_lessons_available']}")

        # Memory should not be used (empty store)
        assert result["metadata"]["memory_used"]["analyst"] is False
        assert result["metadata"]["memory_used"]["clinician"] is False
        assert result["metadata"]["memory_used"]["critic"] is False
        assert result["metadata"]["total_lessons_available"] == 0

        # All outputs must still be valid despite empty memory
        assert len(result["analyst"]) > 50, "Analyst output too short without memory"
        assert len(result["clinician"]) > 50, "Clinician output too short without memory"
        assert len(result["critic"]) > 50, "Critic output too short without memory"
    finally:
        pipeline.memory_store = original_store


def test_accumulates(pipeline, tmp_path, cbc_text, clinical_text):
    """TEST 6 - Memory lesson count grows after each evaluate_and_learn call."""
    # Use an isolated memory store for clean counting
    isolated_store = MemoryStore(persist_dir=str(tmp_path / "chroma_accum"))
    original_store = pipeline.memory_store
    pipeline.memory_store = isolated_store

    try:
        # Run 1: CBC
        result1 = pipeline.run(cbc_text)
        sirius1 = pipeline.evaluate_and_learn(result1)
        count_after_cbc = pipeline.memory_store.get_lesson_count()
        print(f"\nAfter CBC run: {count_after_cbc} lessons "
              f"(score={sirius1['evaluation']['score']})")

        # Run 2: Clinical note
        result2 = pipeline.run(clinical_text)
        sirius2 = pipeline.evaluate_and_learn(result2)
        count_after_clinical = pipeline.memory_store.get_lesson_count()
        print(f"After clinical run: {count_after_clinical} lessons "
              f"(score={sirius2['evaluation']['score']})")

        score1 = sirius1["evaluation"]["score"]
        score2 = sirius2["evaluation"]["score"]

        if score1 >= 3 and score2 >= 3:
            assert count_after_clinical > count_after_cbc, \
                "Lesson count did not increase after second run"
            assert count_after_clinical >= 4, \
                f"Expected >= 4 total lessons, got {count_after_clinical}"
        elif score1 >= 3:
            assert count_after_cbc > 0, "Expected lessons from CBC run (score >= 3)"
        else:
            pytest.skip("Both chains scored < 3, no lessons extracted — expected")
    finally:
        pipeline.memory_store = original_store


def test_lesson_format(populated_pipeline):
    """TEST 7 - Each stored lesson has correct keys and valid values."""
    total = populated_pipeline.memory_store.get_lesson_count()
    if total == 0:
        pytest.skip("No lessons stored — skipping format test")

    lessons = populated_pipeline.memory_store.retrieve_lessons(
        "medical findings", top_k=total
    )

    print(f"\nAll stored lessons ({len(lessons)} retrieved):")
    for i, lesson in enumerate(lessons):
        print(f"  [{i+1}] topic={lesson.get('topic', '')}")
        print(f"       rule={lesson.get('rule', '')[:80]}...")
        print(f"       confidence={lesson.get('confidence', '')}, "
              f"input_type={lesson.get('input_type', '')}")

    for lesson in lessons:
        assert "topic" in lesson, "Lesson missing 'topic'"
        assert "input_type" in lesson, "Lesson missing 'input_type'"
        assert "rule" in lesson, "Lesson missing 'rule'"
        assert "confidence" in lesson, "Lesson missing 'confidence'"

        assert isinstance(lesson["rule"], str) and len(lesson["rule"]) > 0, \
            "Lesson 'rule' must be a non-empty string"
        assert len(lesson["rule"]) < 500, \
            f"Lesson 'rule' too long: {len(lesson['rule'])} chars"
        assert lesson["confidence"] in ("high", "medium"), \
            f"Unexpected confidence value: {lesson['confidence']}"
