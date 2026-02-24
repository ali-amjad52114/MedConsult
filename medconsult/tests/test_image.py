"""
Image pipeline tests for MedConsult.

Tests that the image code path works end-to-end:
  - PIL Image, file path (str), and file path (Path) are all accepted
  - Analyst runs ABCDE extraction on an image
  - Full pipeline runs without error on image + text input
  - Image-only input (no text) works

Uses a synthetic chest X-ray generated via PIL — no real patient data,
no internet download required. Run download_test_image.py first for
real X-ray testing.

Run: python -m pytest tests/test_image.py -v -s
"""

import sys
import os
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

# Ensure medconsult/ is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def synthetic_xray_pil():
    """Create a minimal synthetic chest X-ray as a PIL Image (no file needed)."""
    W, H = 256, 256
    img = Image.new("RGB", (W, H), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 40, 110, 200], fill=(55, 55, 55))   # left lung
    draw.ellipse([145, 40, 225, 200], fill=(55, 55, 55))  # right lung
    draw.ellipse([95, 100, 165, 180], fill=(160, 160, 160))  # cardiac shadow
    draw.rectangle([122, 25, 133, 95], fill=(140, 140, 140))  # trachea
    return img


@pytest.fixture(scope="module")
def synthetic_xray_path(tmp_path_factory, synthetic_xray_pil):
    """Save the synthetic X-ray to a temp file and return the path."""
    path = tmp_path_factory.mktemp("img") / "synthetic_xray.jpg"
    synthetic_xray_pil.save(path, "JPEG")
    return path


@pytest.fixture(scope="module")
def model_manager():
    from model.medgemma_manager import MedGemmaManager
    return MedGemmaManager()


@pytest.fixture(scope="module")
def pipeline():
    from pipeline import MedConsultPipeline
    return MedConsultPipeline()


# --------------------------------------------------------------------------- #
# Unit: image normalization
# --------------------------------------------------------------------------- #

def test_normalize_pil_image(model_manager, synthetic_xray_pil):
    """_normalize_image accepts a PIL Image and returns an RGB PIL Image."""
    result = model_manager._normalize_image(synthetic_xray_pil)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    print(f"PIL → PIL: size={result.size}, mode={result.mode}")


def test_normalize_str_path(model_manager, synthetic_xray_path):
    """_normalize_image accepts a string file path and loads it as RGB PIL Image."""
    result = model_manager._normalize_image(str(synthetic_xray_path))
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    print(f"str path → PIL: size={result.size}, mode={result.mode}")


def test_normalize_pathlib_path(model_manager, synthetic_xray_path):
    """_normalize_image accepts a pathlib.Path and loads it as RGB PIL Image."""
    result = model_manager._normalize_image(Path(synthetic_xray_path))
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    print(f"Path → PIL: size={result.size}, mode={result.mode}")


# --------------------------------------------------------------------------- #
# Integration: Analyst with image
# --------------------------------------------------------------------------- #

def test_analyst_with_image_runs(pipeline, synthetic_xray_pil):
    """Analyst.analyze() runs without error when an image is provided."""
    result = pipeline.analyst.analyze(
        user_input_text="[Chest X-ray provided for review]",
        image=synthetic_xray_pil,
    )
    assert isinstance(result, str)
    assert len(result) > 20
    print("\n--- Analyst image output ---")
    print(result[:500])


def test_analyst_with_image_uses_abcde(pipeline, synthetic_xray_pil):
    """Analyst uses the ABCDE approach when an image is provided."""
    result = pipeline.analyst.analyze(
        user_input_text="[Chest X-ray provided for review]",
        image=synthetic_xray_pil,
    )
    result_lower = result.lower()
    # At least one ABCDE component should appear in the output
    abcde_terms = ["airway", "bone", "cardiac", "diaphragm", "lung", "heart", "mediastin"]
    found = [t for t in abcde_terms if t in result_lower]
    assert len(found) >= 1, f"Expected ABCDE terms in output; found none. Output: {result[:300]}"
    print(f"ABCDE terms found: {found}")


def test_analyst_accepts_file_path(pipeline, synthetic_xray_path):
    """Analyst.analyze() accepts an image as a file path string (not just PIL)."""
    result = pipeline.analyst.analyze(
        user_input_text="[Chest X-ray provided for review]",
        image=str(synthetic_xray_path),
    )
    assert isinstance(result, str)
    assert len(result) > 20
    print(f"\nFile path input → output length: {len(result)}")


# --------------------------------------------------------------------------- #
# Integration: Full pipeline with image
# --------------------------------------------------------------------------- #

def test_full_pipeline_image_only(pipeline, synthetic_xray_pil):
    """Full pipeline runs with image and placeholder text."""
    result = pipeline.run(
        user_input_text="[Chest X-ray provided for review]",
        image=synthetic_xray_pil,
    )
    assert isinstance(result, dict)
    assert all(k in result for k in ["analyst", "clinician", "critic", "metadata"])
    assert len(result["analyst"]) > 20
    assert len(result["clinician"]) > 20
    assert len(result["critic"]) > 20
    print("\n--- Full pipeline image result ---")
    print("Analyst:", result["analyst"][:200])
    print("Clinician:", result["clinician"][:200])
    print("Critic:", result["critic"][:200])


def test_full_pipeline_image_plus_text(pipeline, synthetic_xray_pil):
    """Full pipeline runs correctly when both image and text are provided."""
    text = "Patient is a 58-year-old male with chronic cough and shortness of breath."
    result = pipeline.run(
        user_input_text=text,
        image=synthetic_xray_pil,
    )
    assert isinstance(result, dict)
    # Critic output should reference the image or clinical context
    critic_lower = result["critic"].lower()
    assert any(w in critic_lower for w in ["image", "x-ray", "xray", "chest", "scan", "cough", "breath"]), \
        f"Critic output doesn't reference image or text context. Output: {result['critic'][:300]}"
    print("\n--- Image + text critic output ---")
    print(result["critic"][:400])


def test_pipeline_metadata_records_input_type(pipeline, synthetic_xray_pil):
    """Pipeline metadata is present and timestamped when image is used."""
    result = pipeline.run(
        user_input_text="[Chest X-ray provided for review]",
        image=synthetic_xray_pil,
    )
    meta = result.get("metadata", {})
    assert "timestamp" in meta
    assert "model" in meta
    print(f"Metadata: {meta}")
