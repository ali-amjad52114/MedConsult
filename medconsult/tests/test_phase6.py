"""
Phase 6: Gradio Web Interface tests
"""

import os
import pytest
from unittest.mock import MagicMock, patch

# Import app components (avoids heavy pipeline/chromadb until needed)
from app import _build_ui, _process, launch_ui, _get_lesson_count


@patch("app._get_lesson_count", return_value=0)
def test_initializes(MockGetCount):
    """TEST 1 - App initializes without exception."""
    # Avoid pipeline/chromadb load during UI build
    demo = _build_ui()
    assert demo is not None


@patch("app._get_lesson_count", return_value=0)
def test_empty(MockGetCount):
    """TEST 3 - Empty input returns error message."""
    # _process returns early; badge uses _get_lesson_count
    tab1, tab2, tab3, tab4, tab5, badge = _process("", None, progress_fn=None)
    assert "Please provide" in tab1 or "provide" in tab1.lower()
    assert tab2 == ""
    assert tab3 == ""


@patch("app._get_pipeline")
def test_long(MockGetPipeline):
    """TEST 4 - Long input does not crash."""
    long_text = "Glucose: 100 mg/dL. " * 500  # ~10k chars
    mock_pipeline = MagicMock()
    mock_pipeline.memory_store.get_lesson_count.return_value = 0
    mock_pipeline.run_with_timing.return_value = {
        "analyst": "Analyst output",
        "clinician": "Clinician output",
        "critic": "Critic output",
        "metadata": {"memory_used": {}},
        "timings": {"analyst": 1, "clinician": 1, "critic": 1, "total": 3},
    }
    mock_pipeline.evaluate_and_learn.return_value = {
        "evaluation": {"score": 4, "raw_evaluation": "Good."},
        "augmented": False,
        "lessons_extracted": 0,
        "total_lessons": 0,
        "lessons_injected": [],
    }
    MockGetPipeline.return_value = mock_pipeline

    tab1, tab2, tab3, tab4, tab5, badge = _process(long_text, None, progress_fn=None)

    assert len(tab1) > 0
    assert len(tab2) > 0
    assert len(tab3) > 0
    assert "timing" in tab4 or "total" in tab4


@patch("app._get_pipeline")
def test_processes(MockGetPipeline):
    """TEST 2 - Processes text end to end; all tabs have content."""
    mock_pipeline = MagicMock()
    mock_pipeline.memory_store.get_lesson_count.return_value = 1
    mock_pipeline.run_with_timing.return_value = {
        "analyst": "Analyst extraction here",
        "clinician": "Clinician interpretation here",
        "critic": "Critic patient summary here",
        "metadata": {"memory_used": {}},
        "timings": {"analyst": 1, "clinician": 1, "critic": 1, "total": 3},
    }
    mock_pipeline.evaluate_and_learn.return_value = {
        "evaluation": {"score": 4, "raw_evaluation": "Quality assessment text."},
        "augmented": False,
        "lessons_extracted": 1,
        "total_lessons": 1,
        "lessons_injected": [],
    }
    MockGetPipeline.return_value = mock_pipeline

    tab1, tab2, tab3, tab4, tab5, badge = _process(
        "CBC: WBC 12.8, Hgb 11.2", None, progress_fn=None
    )

    assert "Critic" in tab1 or "patient" in tab1.lower() or "summary" in tab1.lower()
    assert "Analyst" in tab2 or "extraction" in tab2
    assert "Clinician" in tab3 or "interpretation" in tab3
    assert "timing" in tab4
    assert "4" in tab5 or "5" in tab5
    assert len(tab2) > 0


@patch("app._get_pipeline")
def test_sirius_tab(MockGetPipeline):
    """TEST 5 - SiriuS tab receives evaluation (score visible)."""
    mock_pipeline = MagicMock()
    mock_pipeline.memory_store.get_lesson_count.return_value = 0
    mock_pipeline.run_with_timing.return_value = {
        "analyst": "Analyst output",
        "clinician": "Clinician output",
        "critic": "Critic output",
        "metadata": {"memory_used": {}},
        "timings": {"total": 1},
    }
    mock_pipeline.evaluate_and_learn.return_value = {
        "evaluation": {"score": 5, "raw_evaluation": "Excellent extraction and interpretation."},
        "augmented": False,
        "lessons_extracted": 0,
        "total_lessons": 0,
        "lessons_injected": [],
    }
    MockGetPipeline.return_value = mock_pipeline

    tab1, tab2, tab3, tab4, tab5, badge = _process(
        "Glucose: 100 mg/dL", None, progress_fn=None
    )

    assert "5" in tab5, "SiriuS tab should show quality score"
    assert "/5" in tab5 or "5/5" in tab5 or "Score" in tab5


@patch("app._get_pipeline")
def test_sirius_safe(MockGetPipeline):
    """TEST 6 - SiriuS failure does NOT block user results."""
    mock_pipeline = MagicMock()
    mock_pipeline.memory_store.get_lesson_count.return_value = 0
    mock_pipeline.run_with_timing.return_value = {
        "analyst": "Analyst extraction",
        "clinician": "Clinician interpretation",
        "critic": "Patient summary from Critic",
        "metadata": {"memory_used": {}},
        "timings": {"total": 1},
    }
    mock_pipeline.evaluate_and_learn.side_effect = Exception("Cloud API down")
    MockGetPipeline.return_value = mock_pipeline

    tab1, tab2, tab3, tab4, tab5, badge = _process(
        "CBC: WBC 10", None, progress_fn=None
    )

    # User still gets pipeline results
    assert "Patient" in tab1 or "summary" in tab1.lower() or "Critic" in tab1
    assert len(tab2) > 0
    assert len(tab3) > 0
    assert "unavailable" in tab5.lower() or "complete" in tab5.lower() or "valid" in tab5.lower()


@patch("app._get_pipeline")
def test_lesson_badge(MockGetPipeline):
    """TEST 7 - Lesson count badge updates after analysis."""
    mock_pipeline = MagicMock()
    mock_pipeline.memory_store.get_lesson_count.return_value = 5
    mock_pipeline.run_with_timing.return_value = {
        "analyst": "Analyst",
        "clinician": "Clinician",
        "critic": "Critic",
        "metadata": {"memory_used": {}},
        "timings": {"total": 1},
    }
    mock_pipeline.evaluate_and_learn.return_value = {
        "evaluation": {"score": 4, "raw_evaluation": "Suggested Improvements: Improve hemoglobin extraction."},
        "augmented": False,
        "lessons_extracted": 1,
        "total_lessons": 5,
        "lessons_injected": [],
    }
    MockGetPipeline.return_value = mock_pipeline

    tab1, tab2, tab3, tab4, tab5, badge = _process(
        "Hgb 11.2 g/dL", None, progress_fn=None
    )

    assert "lessons" in badge.lower() or "🧠" in badge
    assert any(c.isdigit() for c in badge), "Badge should show a number"
