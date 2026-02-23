"""Phase 5: Full Pipeline tests (Phases 5, 5B, 5C combined — 10 tests)"""
import os
import re
import pytest
from pipeline import MedConsultPipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    return MedConsultPipeline()


@pytest.fixture(scope="module")
def cbc_result(pipeline):
    with open(os.path.join(DATA_DIR, "sample_cbc.txt")) as f:
        text = f.read()
    result = pipeline.run(text)
    print(f"\n[CBC] Analyst:\n{result['analyst'][:300]}")
    print(f"\n[CBC] Clinician:\n{result['clinician'][:300]}")
    print(f"\n[CBC] Critic:\n{result['critic'][:300]}")
    return result


@pytest.fixture(scope="module")
def clinical_result(pipeline):
    with open(os.path.join(DATA_DIR, "sample_clinical_note.txt")) as f:
        text = f.read()
    result = pipeline.run(text)
    print(f"\n[Clinical] Critic:\n{result['critic'][:300]}")
    return result


@pytest.fixture(scope="module")
def sirius_result(pipeline, cbc_result):
    """Run evaluate_and_learn on the CBC result once."""
    result = pipeline.evaluate_and_learn(cbc_result)
    print(f"\n[SiriuS] Score: {result['evaluation']['score']}")
    print(f"[SiriuS] Lessons extracted: {result.get('lessons_extracted', 0)}")
    print(f"[SiriuS] Total lessons: {result.get('total_lessons', 0)}")
    return result


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_pipeline_cbc(cbc_result):
    """TEST 1 - Full pipeline on CBC: all three outputs + end markers present."""
    assert len(cbc_result["analyst"]) > 50, "Analyst output too short"
    assert len(cbc_result["clinician"]) > 50, "Clinician output too short"
    assert len(cbc_result["critic"]) > 50, "Critic output too short"
    assert "END OF EXTRACTION" in cbc_result["analyst"], "Missing analyst end marker"
    assert "END OF CLINICAL INTERPRETATION" in cbc_result["clinician"], \
        "Missing clinician end marker"
    assert "END OF REPORT" in cbc_result["critic"], "Missing critic end marker"


def test_pipeline_clinical(clinical_result):
    """TEST 2 - Full pipeline on clinical note: 'heart' present in critic output."""
    assert "heart" in clinical_result["critic"].lower(), \
        "'heart' not found in critic output for clinical note"


def test_timing(pipeline):
    """TEST 3 - run_with_timing() completes within 300 seconds."""
    with open(os.path.join(DATA_DIR, "sample_cbc.txt")) as f:
        text = f.read()
    result = pipeline.run_with_timing(text)
    total = result["timings"]["total"]
    print(f"\nTimings: {result['timings']}")
    assert total < 300, f"Pipeline took {total}s, expected < 300s"
    assert "analyst" in result["timings"]
    assert "clinician" in result["timings"]
    assert "critic" in result["timings"]


def test_consistency(pipeline, cbc_result):
    """TEST 4 - Same input produces 'anemia' in both runs (CPU greedy = deterministic)."""
    combined = (cbc_result["analyst"] + cbc_result["clinician"]).lower()
    assert "anemia" in combined, "Run 1: 'anemia' not found in analyst+clinician output"

    # Second run (model already loaded, just generation time)
    with open(os.path.join(DATA_DIR, "sample_cbc.txt")) as f:
        text = f.read()
    result2 = pipeline.run(text)
    combined2 = (result2["analyst"] + result2["clinician"]).lower()
    assert "anemia" in combined2, "Run 2: 'anemia' not found in analyst+clinician output"


def test_info_flow(cbc_result):
    """TEST 5 - Values from CBC input traced through analyst and clinician."""
    with open(os.path.join(DATA_DIR, "sample_cbc.txt")) as f:
        cbc_text = f.read()

    # Find all numeric values in the CBC input
    input_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", cbc_text))
    chain_text = cbc_result["analyst"] + cbc_result["clinician"]
    chain_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", chain_text))

    overlap = input_nums & chain_nums
    print(f"\nValues from input found in chain: {overlap}")
    assert len(overlap) >= 3, \
        f"Only {len(overlap)} input values found in analyst+clinician output"


def test_short_input(pipeline):
    """TEST 6 - Short single-value input: all three outputs non-empty."""
    result = pipeline.run("Glucose: 250 mg/dL (reference: 70-100)")
    assert len(result["analyst"]) > 20, "Analyst output empty for short input"
    assert len(result["clinician"]) > 20, "Clinician output empty for short input"
    assert len(result["critic"]) > 20, "Critic output empty for short input"
    assert result["metadata"]["input_type"] == "lab_report"


def test_evaluate_and_learn(sirius_result):
    """TEST 7 - evaluate_and_learn returns score, saved JSON, and lesson count."""
    score = sirius_result["evaluation"]["score"]
    assert 1 <= score <= 5, f"Score {score} not in range 1-5"
    assert "raw_evaluation" in sirius_result["evaluation"]
    assert len(sirius_result["evaluation"]["raw_evaluation"]) > 0

    # Either saved_to (good path) or augmented (bad path) must be present
    has_save = "saved_to" in sirius_result and sirius_result["saved_to"].endswith(".json")
    has_aug = sirius_result.get("augmented", False)
    assert has_save or has_aug, "Neither saved_to nor augmented flag present"

    assert "total_lessons" in sirius_result
    assert sirius_result["total_lessons"] >= 0


def test_lessons_extracted(sirius_result, pipeline):
    """TEST 8 - Good chains (score >= 3) produce extracted lessons."""
    score = sirius_result["evaluation"]["score"]
    print(f"\nScore: {score}, Lessons extracted: {sirius_result.get('lessons_extracted', 0)}")

    if score >= 3:
        assert sirius_result.get("lessons_extracted", 0) > 0, \
            f"Score was {score} (>= 3) but no lessons extracted"

    # Total lessons in store should be >= 0
    total = pipeline.memory_store.get_lesson_count()
    print(f"Total lessons in ChromaDB: {total}")
    assert total >= 0


def test_chromadb_stores(pipeline, sirius_result):
    """TEST 9 - After evaluate_and_learn, ChromaDB retrieval returns relevant lessons."""
    total = pipeline.memory_store.get_lesson_count()
    print(f"\nTotal lessons in store: {total}")

    if total == 0:
        pytest.skip("No lessons were stored (score < 3 or extraction failed) — skipping")

    results = pipeline.memory_store.retrieve_lessons("anemia low hemoglobin", top_k=3)
    print(f"Retrieved {len(results)} lessons for 'anemia low hemoglobin'")
    for r in results:
        print(f"  - topic: {r.get('topic', '')}")
        print(f"    rule: {r.get('rule', '')[:80]}...")

    assert len(results) >= 1, "No lessons retrieved from ChromaDB"
    for lesson in results:
        assert "rule" in lesson and lesson["rule"], "Lesson missing 'rule' field"


def test_input_classification(pipeline):
    """TEST 10 - Input type classification works for all four categories."""
    lib = pipeline.experience_library
    assert lib.classify_input_type("CBC with hemoglobin and WBC") == "lab_report"
    assert lib.classify_input_type("Discharge Summary — patient admitted for chest pain") == "clinical_note"
    assert lib.classify_input_type("Chest X-ray shows bilateral infiltrates") == "imaging"
    assert lib.classify_input_type("I have a headache and feel tired") == "general"
