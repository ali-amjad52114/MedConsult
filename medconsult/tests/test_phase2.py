import pytest
import os
from agents.analyst import AnalystAgent
from model.model_manager import ModelManager

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture
def analyst():
    """AnalystAgent with MedGemma model manager."""
    return AnalystAgent(ModelManager())


def test_extracts_lab_values(analyst):
    """TEST 1 - Extracts lab values from CBC report."""
    with open(os.path.join(DATA_DIR, "sample_cbc.txt"), "r") as f:
        cbc_text = f.read()

    response = analyst.analyze(cbc_text)

    assert "12.8" in response, "Key value 12.8 (WBC) not found in response"
    assert "HIGH" in response, "No HIGH flags found in response"
    assert "LOW" in response, "No LOW flags found in response"
    assert "NORMAL" in response, "No NORMAL values found in response"
    assert "END OF EXTRACTION" in response, "Missing END OF EXTRACTION marker"

    print(f"\nLab values response:\n{response}")


def test_extracts_clinical_note(analyst):
    """TEST 2 - Extracts clinical note: age, diagnosis, meds, vitals."""
    with open(os.path.join(DATA_DIR, "sample_clinical_note.txt"), "r") as f:
        clinical_note_text = f.read()

    response = analyst.analyze(clinical_note_text)

    assert "67" in response, "Patient age 67 not found in response"
    assert "heart failure" in response.lower(), "'heart failure' not found in response"
    assert "furosemide" in response.lower(), "'furosemide' not found in response"
    assert "162" in response, "BP value 162 not found in response"
    assert "END OF EXTRACTION" in response, "Missing END OF EXTRACTION marker"

    print(f"\nClinical note response:\n{response}")


def test_no_interpretation(analyst):
    """TEST 3 - Does NOT interpret: no suggest/recommend/diagnosis."""
    with open(os.path.join(DATA_DIR, "sample_cbc.txt"), "r") as f:
        cbc_text = f.read()

    response = analyst.analyze(cbc_text)
    response_lower = response.lower()

    assert "suggest" not in response_lower, "Response contains 'suggest' — Analyst is interpreting"
    assert "recommend" not in response_lower, "Response contains 'recommend' — Analyst is interpreting"
    assert "diagnosis" not in response_lower, "Response contains 'diagnosis' — Analyst is interpreting"

    print(f"\nNo-interpretation response:\n{response}")


def test_short_input(analyst):
    """TEST 4 - Extracts single value from short input."""
    short_input = "Glucose: 250 mg/dL (reference: 70-100)"

    response = analyst.analyze(short_input)

    assert "250" in response, "Glucose value 250 not found in response"
    assert any(
        word in response.upper() for word in ["HIGH", "ABOVE", "ELEVATED"]
    ), "No flag for out-of-range glucose found in response"

    print(f"\nShort input response:\n{response}")
