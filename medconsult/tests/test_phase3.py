import os
import pytest
import re
from agents.analyst import AnalystAgent
from agents.clinician import ClinicianAgent
from model.model_manager import ModelManager

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

@pytest.fixture(scope="module")
def shared_model_manager():
    return ModelManager()

@pytest.fixture(scope="module")
def analyst(shared_model_manager):
    return AnalystAgent(shared_model_manager)


@pytest.fixture(scope="module")
def clinician(shared_model_manager):
    return ClinicianAgent(model_manager=shared_model_manager)


def test_identifies_anemia(analyst, clinician):
    """TEST 1 - Clinician identifies anemia pattern from CBC"""
    with open(os.path.join(DATA_DIR, "sample_cbc.txt"), "r") as f:
        cbc_text = f.read()

    analyst_output = analyst.analyze(cbc_text)
    response = clinician.interpret(cbc_text, analyst_output)

    print(f"\nResponse:\n{response}")

    response_lower = response.lower()
    
    # Assert response contains "anemia" or synonyms
    has_anemia = "anemia" in response_lower or "low red blood cell" in response_lower or "low hemoglobin pattern" in response_lower or "decreased rbc" in response_lower
    assert has_anemia, "Did not identify anemia pattern"

    has_urgency = "ROUTINE" in response or "WORTH DISCUSSING" in response or "NEEDS PROMPT ATTENTION" in response
    assert has_urgency, "Response missing proper urgency rating"


def test_identifies_heart_failure(analyst, clinician):
    """TEST 2 - Clinician identifies heart failure from clinical note"""
    with open(os.path.join(DATA_DIR, "sample_clinical_note.txt"), "r") as f:
        clinical_note_text = f.read()

    analyst_output = analyst.analyze(clinical_note_text)
    response = clinician.interpret(clinical_note_text, analyst_output)

    print(f"\nResponse:\n{response}")

    response_lower = response.lower()
    assert "heart failure" in response_lower, "Did not identify heart failure"
    assert "furosemide" in response_lower or "diuretic" in response_lower, "Did not connect medication non-compliance"
    assert "fluid" in response_lower or "edema" in response_lower or "congestion" in response_lower, "Did not mention fluid/edema/congestion"


def test_considers_together(analyst, clinician):
    """TEST 3 - Clinician considers findings TOGETHER, not in isolation"""
    with open(os.path.join(DATA_DIR, "sample_cbc.txt"), "r") as f:
        cbc_text = f.read()

    analyst_output = analyst.analyze(cbc_text)
    response = clinician.interpret(cbc_text, analyst_output)

    print(f"\nResponse:\n{response}")
    
    response_lower = response.lower()
    sentences = response_lower.replace('\\n', '. ').split('.')
    found_together = False
    for sentence in sentences:
        if "hemoglobin" in sentence and ("rbc" in sentence or "hematocrit" in sentence or "red blood" in sentence or "red blood cell" in sentence):
            found_together = True
            break
            
    assert found_together, "Did not find findings considered together in the same sentence"


def test_cites_values(analyst, clinician):
    """TEST 4 - Clinician cites specific values"""
    with open(os.path.join(DATA_DIR, "sample_cbc.txt"), "r") as f:
        cbc_text = f.read()

    analyst_output = analyst.analyze(cbc_text)
    response = clinician.interpret(cbc_text, analyst_output)

    print(f"\nResponse:\n{response}")
    
    numbers_in_response = re.findall(r'\\b\\d+(?:\\.\\d+)?\\b', response)
    numbers_in_cbc = set(re.findall(r'\\b\\d+(?:\\.\\d+)?\\b', cbc_text))
    
    overlap = set()
    for num in numbers_in_response:
        if num in numbers_in_cbc:
            overlap.add(num)
            
    assert len(overlap) >= 3, f"Response didn't cite enough numeric values. Overlap: {overlap}"


def test_urgency_rating(analyst, clinician):
    """TEST 5 - Clinician provides urgency rating"""
    with open(os.path.join(DATA_DIR, "sample_cbc.txt"), "r") as f:
        cbc_text = f.read()

    analyst_output = analyst.analyze(cbc_text)
    response = clinician.interpret(cbc_text, analyst_output)

    print(f"\nResponse:\n{response}")
    
    urgency_levels = ["ROUTINE", "WORTH DISCUSSING", "NEEDS PROMPT ATTENTION"]
    count = sum(1 for level in urgency_levels if level in response)
    
    assert count == 1, f"Response should contain exactly one urgency keyword, found {count}"
