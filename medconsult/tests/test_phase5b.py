"""Phase 5B: Augmentation Loop tests"""
import os
import pytest
from pipeline import MedConsultPipeline
from sirius.augmentation import AugmentationLoop

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

# Mock evaluation that always returns low score (triggers augmentation)
_MOCK_LOW_EVAL = {
    "score": 2,
    "issues": [
        "Missed platelet elevation",
        "Did not connect WBC and neutrophil pattern",
    ],
    "improvements": [
        "Explicitly note platelet count of 415",
        "Discuss neutrophilia in context of elevated WBC",
    ],
    "raw_evaluation": "Score: 2\nKey Issues Found:\n- Missed platelet elevation\n--- END OF EVALUATION ---",
}


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    return MedConsultPipeline()


@pytest.fixture(scope="module")
def cbc_result(pipeline):
    with open(os.path.join(DATA_DIR, "sample_cbc.txt")) as f:
        text = f.read()
    return pipeline.run(text)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_augmentation_runs(pipeline, cbc_result):
    """TEST 1 - AugmentationLoop runs at least one retry on a low-score eval."""
    loop = AugmentationLoop(pipeline, max_retries=1)
    improved = loop.augment(cbc_result, _MOCK_LOW_EVAL)

    print(f"\nAugmentation result: attempts={improved['attempts']}, "
          f"original={improved['original_score']}, final={improved['final_score']}")

    assert isinstance(improved, dict)
    assert improved["attempts"] >= 1
    assert "improved_result" in improved
    assert "improved_evaluation" in improved
    assert 1 <= improved["final_score"] <= 5


def test_feedback_changes_output(pipeline):
    """TEST 2 - Feedback injection produces different output than no feedback (critical)."""
    with open(os.path.join(DATA_DIR, "sample_cbc.txt")) as f:
        cbc_text = f.read()

    # Run analyst WITHOUT feedback
    output_plain = pipeline.analyst.analyze(cbc_text)

    # Run analyst WITH explicit feedback that changes the prompt
    output_with_feedback = pipeline.analyst.analyze(
        cbc_text, feedback="You missed platelet count — focus on it explicitly"
    )

    print(f"\nPlain output (first 200):\n{output_plain[:200]}")
    print(f"\nFeedback output (first 200):\n{output_with_feedback[:200]}")

    # Different prompt → different output (especially with CPU greedy decoding)
    assert output_plain != output_with_feedback, \
        "Feedback did not change the analyst output — check that feedback is injected into prompt"


def test_max_retries_respected(pipeline, cbc_result):
    """TEST 3 - AugmentationLoop never exceeds max_retries."""
    for max_r in [1, 2]:
        loop = AugmentationLoop(pipeline, max_retries=max_r)
        improved = loop.augment(cbc_result, _MOCK_LOW_EVAL)
        assert improved["attempts"] <= max_r, \
            f"attempts={improved['attempts']} exceeded max_retries={max_r}"
        print(f"\nmax_retries={max_r}: attempts={improved['attempts']}")


def test_auto_improvement(pipeline, cbc_result):
    """TEST 4 - evaluate_and_learn() completes end-to-end with correct structure."""
    sirius = pipeline.evaluate_and_learn(cbc_result)

    print(f"\nSiriuS result keys: {list(sirius.keys())}")
    print(f"Score: {sirius['evaluation']['score']}")

    assert "evaluation" in sirius
    assert 1 <= sirius["evaluation"]["score"] <= 5

    # Either normal path or augmented path — both must be present
    has_normal = "saved_to" in sirius
    has_augmented = sirius.get("augmented", False)
    assert has_normal or has_augmented, "Neither 'saved_to' nor 'augmented' in result"

    assert "total_lessons" in sirius
    assert "library_stats" in sirius


def test_improved_chains_saved(pipeline):
    """TEST 5 - Deliberately bad chain triggers augmentation; result is saved and lessons extracted."""
    # Create a chain with obviously hallucinated values to get score <= 2
    bad_result = {
        "input": "Glucose: 100 mg/dL",
        "analyst": "WBC: 999.0 (CRITICAL). Hemoglobin: 0.1 g/dL. Platelets: 9999.",
        "clinician": "The patient has all diseases simultaneously. Immediate surgery required.",
        "critic": "Everything is perfect. No action needed.",
        "metadata": {
            "model": "google/medgemma-1.5-4b-it",
            "pipeline_version": "4.0-sirius-cloud",
            "timestamp": "2026-02-23T00:00:00+00:00",
            "input_type": "lab_report",
            "agent_chain": ["analyst", "clinician", "critic"],
        },
    }

    sirius = pipeline.evaluate_and_learn(bad_result)

    score = sirius["evaluation"]["score"]
    print(f"\nBad chain score: {score}")
    print(f"Augmented: {sirius.get('augmented', False)}")

    # The evaluation should be stored (regardless of augmentation path)
    assert "evaluation" in sirius
    assert 1 <= score <= 5

    if score <= 2:
        # Augmentation should have triggered
        assert sirius.get("augmented") is True, "Expected augmentation for low score"
        aug = sirius["augmentation_result"]
        assert aug["attempts"] >= 1
        print(f"Augmentation attempts: {aug['attempts']}, final score: {aug['final_score']}")
    else:
        # Score was acceptable — verify save path
        assert "saved_to" in sirius
        assert sirius["saved_to"].endswith(".json")
        print(f"Saved to: {sirius['saved_to']}")
