"""Phase 4: Critic & Communicator Agent tests"""
import os
import pytest
from agents.analyst import AnalystAgent
from agents.clinician import ClinicianAgent
from agents.critic import CriticAgent
from model.model_manager import ModelManager

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture(scope="module")
def shared_model():
    return ModelManager()


@pytest.fixture(scope="module")
def cbc_report(shared_model):
    """Run full pipeline on CBC, return critic report."""
    with open(os.path.join(DATA_DIR, "sample_cbc.txt")) as f:
        text = f.read()
    analyst_out = AnalystAgent(shared_model).analyze(text)
    clinician_out = ClinicianAgent(shared_model).interpret(text, analyst_out)
    report = CriticAgent(shared_model).review_and_communicate(text, analyst_out, clinician_out)
    print(f"\nCBC Critic Report:\n{report}")
    return report


@pytest.fixture(scope="module")
def clinical_report(shared_model):
    """Run full pipeline on clinical note, return critic report."""
    with open(os.path.join(DATA_DIR, "sample_clinical_note.txt")) as f:
        text = f.read()
    analyst_out = AnalystAgent(shared_model).analyze(text)
    clinician_out = ClinicianAgent(shared_model).interpret(text, analyst_out)
    report = CriticAgent(shared_model).review_and_communicate(text, analyst_out, clinician_out)
    print(f"\nClinical Critic Report:\n{report}")
    return report


def test_structure(cbc_report):
    """TEST 1 - Report has all required section headers and markers."""
    assert "WHAT WAS CHECKED" in cbc_report, "Missing 'WHAT WAS CHECKED' section"
    assert "QUESTIONS TO ASK" in cbc_report, "Missing 'QUESTIONS TO ASK' section"
    assert "DISCLAIMER" in cbc_report.upper(), "Missing disclaimer section"
    assert "END OF REPORT" in cbc_report, "Missing 'END OF REPORT' marker"


def test_clinical_summary(clinical_report):
    """TEST 2 - Clinical note report mentions heart, doctor, and disclaimer."""
    report_lower = clinical_report.lower()
    assert "heart" in report_lower, "Missing heart-related content in clinical summary"
    assert "doctor" in report_lower, "Missing reference to doctor in clinical summary"
    assert "DISCLAIMER" in clinical_report.upper(), "Missing disclaimer in clinical summary"


def test_plain_language(cbc_report):
    """TEST 3 - Medical jargon has parenthetical explanation nearby."""
    report_lower = cbc_report.lower()
    jargon_terms = ["hemoglobin", "hematocrit", "wbc", "platelets", "neutrophils"]
    found_explained = False
    for term in jargon_terms:
        idx = report_lower.find(term)
        if idx != -1:
            # Check for a '(' within 60 chars before or after the term
            start = max(0, idx - 60)
            end = min(len(cbc_report), idx + 60)
            nearby = cbc_report[start:end]
            if "(" in nearby:
                found_explained = True
                break
    assert found_explained, "No jargon term found with a nearby parenthetical explanation"


def test_patient_questions(cbc_report):
    """TEST 4 - Report contains 3+ patient-perspective questions."""
    question_count = cbc_report.count("?")
    assert question_count >= 3, f"Expected 3+ questions, found {question_count}"


def test_critic_reviews(cbc_report):
    """TEST 5 - Report contains critical review language."""
    report_lower = cbc_report.lower()
    review_terms = [
        "review", "assess", "issues", "interpretation",
        "clinician", "significant", "no significant",
    ]
    has_review_language = any(term in report_lower for term in review_terms)
    assert has_review_language, "No critical review language found in report"


def test_disclaimer(cbc_report):
    """TEST 6 - Report contains proper safety disclaimer."""
    report_lower = cbc_report.lower()
    has_ai_ref = "ai assistant" in report_lower or "not a medical diagnosis" in report_lower
    assert has_ai_ref, "Missing 'AI assistant' or 'not a medical diagnosis' in disclaimer"
    has_provider_ref = "healthcare provider" in report_lower or "doctor" in report_lower
    assert has_provider_ref, "Missing 'healthcare provider' or 'doctor' in disclaimer"
