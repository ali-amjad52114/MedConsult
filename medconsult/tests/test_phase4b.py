import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pytest
from agents.analyst import AnalystAgent
from agents.clinician import ClinicianAgent
from agents.critic import CriticAgent
from agents.evaluator import EvaluatorAgent
from model.model_manager import ModelManager
from model.cloud_manager import CloudManager

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

@pytest.fixture(scope="module")
def shared_model_manager():
    return ModelManager()

@pytest.fixture(scope="module")
def cloud_manager():
    return CloudManager()

@pytest.fixture(scope="module")
def analyst(shared_model_manager):
    return AnalystAgent(shared_model_manager)

@pytest.fixture(scope="module")
def clinician(shared_model_manager):
    return ClinicianAgent(model_manager=shared_model_manager)

@pytest.fixture(scope="module")
def critic(shared_model_manager):
    return CriticAgent(model_manager=shared_model_manager)

@pytest.fixture(scope="module")
def evaluator(cloud_manager):
    return EvaluatorAgent(cloud_manager)

# Cache chains to speed up tests, since MedGemma is slow
_cbc_chain = {}
_clinical_chain = {}

def get_cbc_chain(analyst, clinician, critic):
    if not _cbc_chain:
        with open(os.path.join(DATA_DIR, "sample_cbc.txt"), "r") as f:
            _cbc_chain["input"] = f.read()
            
        print("\n[CBC] Running Analyst...")
        _cbc_chain["analyst"] = analyst.analyze(_cbc_chain["input"])
        
        print("[CBC] Running Clinician...")
        _cbc_chain["clinician"] = clinician.interpret(_cbc_chain["input"], _cbc_chain["analyst"])
        
        print("[CBC] Running Critic...")
        _cbc_chain["critic"] = critic.review_and_communicate(_cbc_chain["input"], _cbc_chain["analyst"], _cbc_chain["clinician"])
    return _cbc_chain

def get_clinical_chain(analyst, clinician, critic):
    if not _clinical_chain:
        with open(os.path.join(DATA_DIR, "sample_clinical_note.txt"), "r") as f:
            _clinical_chain["input"] = f.read()
            
        print("\n[Clinical] Running Analyst...")
        _clinical_chain["analyst"] = analyst.analyze(_clinical_chain["input"])
        
        print("[Clinical] Running Clinician...")
        _clinical_chain["clinician"] = clinician.interpret(_clinical_chain["input"], _clinical_chain["analyst"])
        
        print("[Clinical] Running Critic...")
        _clinical_chain["critic"] = critic.review_and_communicate(_clinical_chain["input"], _clinical_chain["analyst"], _clinical_chain["clinician"])
    return _clinical_chain

def test_valid_score_cbc(analyst, clinician, critic, evaluator):
    """TEST 1 - Valid score for CBC chain"""
    chain = get_cbc_chain(analyst, clinician, critic)
    
    print("\n[CBC] Running Evaluator...")
    result = evaluator.evaluate(chain["input"], chain["analyst"], chain["clinician"], chain["critic"])
    
    print(f"\nEvaluator Raw:\n{result['raw_evaluation']}")
    assert isinstance(result, dict)
    assert 1 <= result["score"] <= 5
    assert len(result["raw_evaluation"]) > 0
    assert "END OF EVALUATION" in result["raw_evaluation"]

def test_valid_score_clinical(analyst, clinician, critic, evaluator):
    """TEST 2 - Valid score for clinical note"""
    chain = get_clinical_chain(analyst, clinician, critic)
    
    print("\n[Clinical] Running Evaluator...")
    result = evaluator.evaluate(chain["input"], chain["analyst"], chain["clinician"], chain["critic"])
    
    print(f"\nEvaluator Raw:\n{result['raw_evaluation']}")
    assert isinstance(result, dict)
    assert 1 <= result["score"] <= 5
    assert len(result["raw_evaluation"]) > 0

def test_reasonable(analyst, clinician, critic, evaluator):
    """TEST 3 - Reasonable scores for good chain"""
    chain = get_cbc_chain(analyst, clinician, critic)
    result = evaluator.evaluate(chain["input"], chain["analyst"], chain["clinician"], chain["critic"])
    
    assert result["score"] >= 3, f"Score was {result['score']}, expected >= 3"

def test_catches_bad(analyst, clinician, critic, evaluator):
    """TEST 4 - Catches bad input (critical)"""
    chain = get_cbc_chain(analyst, clinician, critic)
    
    # Deliberately fake analyst output with a hallucinated value
    bad_analyst = "Hemoglobin of 45.0 g/dL (CRITICAL HIGH), Platelets 900 (HIGH)"
    
    print("\n[Bad Input] Running Evaluator...")
    result = evaluator.evaluate(chain["input"], bad_analyst, chain["clinician"], chain["critic"])
    
    print(f"\nEvaluator Raw (Bad Input):\n{result['raw_evaluation']}")
    assert result["score"] <= 2, f"Score was {result['score']}, expected <= 2 for hallucinated input"
    assert len(result["issues"]) > 0, "Expected evaluator to find issues"

def test_parsing(analyst, clinician, critic, evaluator):
    """TEST 5 - Parsing robustness"""
    chain = get_cbc_chain(analyst, clinician, critic)
    result = evaluator.evaluate(chain["input"], chain["analyst"], chain["clinician"], chain["critic"])
    
    assert isinstance(result["score"], int), "Score should be parsed as int"
    assert isinstance(result["issues"], list), "Issues should be parsed as list"
    assert isinstance(result["improvements"], list), "Improvements should be parsed as list"
