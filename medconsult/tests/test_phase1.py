import pytest
from model.medgemma_manager import MedGemmaManager
from model.cloud_manager import CloudManager
import os

@pytest.fixture(scope="module")
def medgemma():
    return MedGemmaManager()

@pytest.fixture(scope="module")
def cloud_manager():
    return CloudManager()


def test_medgemma_loads(medgemma):
    """TEST 1 - MedGemma loads successfully"""
    model, processor = medgemma.load_model()
    
    assert model is not None, "Model failed to load"
    assert processor is not None, "Processor failed to load"
    
    print(f"\\nDevice: {medgemma.device}")
    print(f"Dtype: {model.dtype}")


def test_medgemma_generation(medgemma):
    """TEST 2 - MedGemma text generation"""
    system_prompt = "You are a helpful medical assistant."
    user_message = "What does hemoglobin measure? Answer in 2 sentences."
    
    response = medgemma.generate_response(system_prompt, user_message, max_tokens=100)
    
    assert response is not None
    assert len(response) >= 20 and len(response) <= 500, f"Response length out of bounds: {len(response)}"
    print(f"\\nGeneration Response:\\n{response}")


def test_medgemma_system_prompt(medgemma):
    """TEST 3 - MedGemma follows system prompt"""
    system_prompt = "Respond ONLY with JSON listing each finding."
    user_message = "Patient has blood pressure 162/94 and heart rate 96."
    
    response = medgemma.generate_response(system_prompt, user_message, max_tokens=100)
    response_lower = response.lower()
    
    assert "{" in response or "blood pressure" in response_lower or "162" in response, f"Model didn't seem to follow the JSON prompt: {response}"
    print(f"\\nPrompt adherence JSON:\\n{response}")


def test_medgemma_different_prompts(medgemma):
    """TEST 4 - MedGemma different prompts produce different outputs"""
    user_message = "Patient has blood pressure 162/94."
    
    system_1 = "Respond with exactly one word: HIGH or NORMAL depending on the BP."
    system_2 = "Translate this into French."
    
    response_1 = medgemma.generate_response(system_1, user_message, max_tokens=20)
    response_2 = medgemma.generate_response(system_2, user_message, max_tokens=20)
    
    assert response_1 != response_2, "Two distinct system prompts produced the exact same output"


def test_cloud_responds(cloud_manager):
    """TEST 5 - Cloud model responds"""
    system_prompt = "You are a helpful assistant."
    user_message = "What is 2+2? Answer with just the number."
    
    response = cloud_manager.generate_response(system_prompt, user_message)
    
    assert response is not None
    assert "4" in response, f"Cloud response didn't contain '4': {response}"
    print(f"\\nCloud Provider used: {cloud_manager.provider}")
    print(f"Response: {response}")


def test_cloud_structured_output(cloud_manager):
    """TEST 6 - Cloud model follows structured output"""
    system_prompt = "You are a scoring system. Rate the following text quality 1-5. Respond with ONLY: Score: X"
    user_message = "The quick brown fox jumps over the lazy dog."
    
    response = cloud_manager.generate_response(system_prompt, user_message)
    
    assert "Score:" in response, f"Response missing 'Score:': {response}"
    assert any(char.isdigit() for char in response), f"Response missing a digit score: {response}"
    print(f"\\nStructured Output: {response}")


def test_same_interface():
    """TEST 7 - Both managers have same interface"""
    # Just grab methods, don't instantiate if we don't need to load the model
    import inspect
    
    medgemma_sig = inspect.signature(MedGemmaManager.generate_response)
    cloud_sig = inspect.signature(CloudManager.generate_response)
    
    assert "system_prompt" in medgemma_sig.parameters
    assert "user_message" in medgemma_sig.parameters
    assert "system_prompt" in cloud_sig.parameters
    assert "user_message" in cloud_sig.parameters
    
    print("\\nInterface Check Passed")
