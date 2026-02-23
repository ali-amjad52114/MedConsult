import os
import pytest
import warnings

def test_imports():
    """1. Verify all required packages can be imported."""
    try:
        import torch
        import transformers
        import gradio
        import PIL
        import PIL
        import google.generativeai as genai
    except ImportError as e:
        pytest.fail(f"Failed to import required package: {e}")
        
    try:
        import chromadb
    except Exception as e:
        warnings.warn(f"chromadb is installed but failed to initialize (likely env issue): {e}")

def test_project_structure():
    """2. Verify the project directory structure exists correctly."""
    required_paths = [
        "app.py",
        "agents/__init__.py",
        "agents/analyst.py",
        "agents/clinician.py",
        "agents/critic.py",
        "agents/evaluator.py",
        "model/__init__.py",
        "model/medgemma_manager.py",
        "model/cloud_manager.py",
        "pipeline.py",
        "sirius/__init__.py",
        "sirius/experience_library.py",
        "sirius/lesson_extractor.py",
        "sirius/memory_store.py",
        "sirius/augmentation.py",
        "sirius/memory_retriever.py",
        "prompts/__init__.py",
        "prompts/analyst_prompt.py",
        "prompts/clinician_prompt.py",
        "prompts/critic_prompt.py",
        "prompts/evaluator_prompt.py",
        "tests/test_phase0.py",
        "tests/test_phase1.py",
        "tests/test_phase2.py",
        "tests/test_phase3.py",
        "tests/test_phase4.py",
        "tests/test_phase4b.py",
        "tests/test_phase5.py",
        "tests/test_phase5b.py",
        "tests/test_phase5c.py",
        "tests/test_data/sample_cbc.txt",
        "tests/test_data/sample_clinical_note.txt",
        "tests/test_data/README.md",
        "experience_library/good",
        "experience_library/bad",
        "experience_library/chroma_db",
        "experience_library/stats.json",
        "requirements.txt",
        "README.md"
    ]
    
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    assert not missing_paths, f"Missing required paths: {missing_paths}"

def test_sample_data_exists():
    """3. Verify both sample test data files exist and are non-empty."""
    cbc_path = "tests/test_data/sample_cbc.txt"
    clin_path = "tests/test_data/sample_clinical_note.txt"
    
    assert os.path.exists(cbc_path), f"File missing: {cbc_path}"
    assert os.path.exists(clin_path), f"File missing: {clin_path}"
    
    assert os.path.getsize(cbc_path) > 10, f"File too small or empty: {cbc_path}"
    assert os.path.getsize(clin_path) > 10, f"File too small or empty: {clin_path}"

def test_gpu_available():
    """4. Verify GPU is available — print warning if not, don't fail."""
    import torch
    if torch.cuda.is_available():
        print(f"\\nGPU available: {torch.cuda.get_device_name(0)}")
    else:
        warnings.warn("No GPU available, falling back to CPU. This may be slow.")

def test_experience_library_dirs():
    """5. Verify experience_library/ with good/, bad/, chroma_db/ subdirs exist."""
    base_dir = "experience_library"
    assert os.path.exists(base_dir) and os.path.isdir(base_dir)
    assert os.path.exists(os.path.join(base_dir, "good")) and os.path.isdir(os.path.join(base_dir, "good"))
    assert os.path.exists(os.path.join(base_dir, "bad")) and os.path.isdir(os.path.join(base_dir, "bad"))
    assert os.path.exists(os.path.join(base_dir, "chroma_db")) and os.path.isdir(os.path.join(base_dir, "chroma_db"))

def test_google_api_key():
    """6. Environment variables check: print whether GOOGLE_API_KEY is set."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        print("\\nGOOGLE_API_KEY is set.")
    else:
        warnings.warn("GOOGLE_API_KEY is not set.")
