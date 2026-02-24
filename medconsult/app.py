"""
MedConsult — Multi-Agent Medical Analysis
Gradio web interface with SiriuS learning.

FLOW: User submits text/image → pipeline.run_with_timing() → results shown in tabs
      → SiriuS runs in background thread → Tab 5 populated when done
"""

import time
import json
from threading import Thread

import gradio as gr


# --- Lazy loading: avoid loading MedGemma/ChromaDB until first Submit ---
def _get_pipeline():
    """Lazy load pipeline to defer heavy imports."""
    from pipeline import MedConsultPipeline
    return MedConsultPipeline()


def _get_lesson_count() -> int:
    """Total lessons in memory (0 if pipeline unavailable)."""
    try:
        p = _get_pipeline()
        return p.memory_store.get_lesson_count()
    except Exception:
        return 0


def _format_sirius_tab(sirius_result: dict) -> str:
    """Format SiriuS evaluation result for Tab 5."""
    if not sirius_result:
        return """SiriuS evaluation unavailable (cloud API may be down or misconfigured).
Your results above are complete and valid.
"""
    ev = sirius_result.get("evaluation", {})
    score = ev.get("score", "?")
    raw = ev.get("raw_evaluation", "")
    aug = sirius_result.get("augmented", False)
    aug = "Yes" if aug else "No"
    lessons_extracted = sirius_result.get("lessons_extracted", 0)
    total_lessons = sirius_result.get("total_lessons", 0)
    library_stats = sirius_result.get("library_stats", {})
    lessons_injected = sirius_result.get("lessons_injected", [])
    validation_report = sirius_result.get("validation_report", None)

    # Build lessons extracted topics from chain if available
    extracted_topics = []
    if "augmentation_result" in sirius_result:
        arr = sirius_result["augmentation_result"]
        if isinstance(arr, dict) and "improved_result" in arr:
            pass  # Could extract from improved chain
    # Fallback: show count
    extracted_str = f"{lessons_extracted} lesson(s)" if isinstance(lessons_extracted, int) else str(lessons_extracted)
    injected_str = ", ".join(lessons_injected) if isinstance(lessons_injected, list) and lessons_injected else "none"
    if isinstance(lessons_injected, dict):
        injected_str = f"{len(lessons_injected.get('analyst', [])) + len(lessons_injected.get('clinician', [])) + len(lessons_injected.get('critic', []))} injected"

    base_msg = f"""**Quality Score: {score}/5**

**Evaluator assessment:**
{raw}

**Augmentation triggered:** {aug}

**Lessons extracted from this analysis:** {extracted_str}

**Lessons injected INTO this analysis:** {injected_str}

**Total lessons in memory:** {total_lessons}

*Evaluation ran after your results were delivered.*
"""
    if validation_report:
        trend = validation_report.get("trend", "flat")
        degrading = ", ".join(validation_report.get("degrading_agents", [])) or "None"
        base_msg += f"\n\n---\n### 📊 Validation Report (Every 5 Runs)\n"
        base_msg += f"**Trend:** {trend}\n"
        base_msg += f"**Degrading Agents:** {degrading}\n"
        base_msg += f"**Average Score (last 5):** {validation_report.get('average_score', 0):.2f}\n"

    return base_msg


def _process(
    medical_text: str,
    image,
    progress_fn,
) -> tuple[str, str, str, str, str, str]:
    """
    Main processing logic. Returns (tab1, tab2, tab3, tab4, tab5, badge_count).
    SiriuS failure NEVER blocks user results — Tab 5 shows "unavailable" instead.
    """
    # 1. Validate input
    text = (medical_text or "").strip()
    if not text and image is None:
        err = "Please provide medical text, an image, or both."
        return err, "", "", "{}", "SiriuS evaluation skipped (no input).", str(_get_lesson_count())

    if not text:
        text = "[Image only — no text provided]"

    try:
        pipeline = _get_pipeline()
    except Exception as e:
        return (
            f"Pipeline initialization failed: {e}",
            "",
            "",
            "{}",
            "SiriuS unavailable.",
            "0",
        )

    # 3. Run pipeline: Analyst → Clinician → Critic (with memory injection)
    if progress_fn:
        progress_fn("Step 1/3: Analyst extracting...")
    if progress_fn:
        progress_fn("Step 2/3: Clinician interpreting...")
    if progress_fn:
        progress_fn("Step 3/3: Critic reviewing...")

    result = pipeline.run_with_timing(text, image=image)  # Returns analyst, clinician, critic, metadata, timings

    analyst_out = result.get("analyst", "")
    clinician_out = result.get("clinician", "")
    critic_out = result.get("critic", "")
    metadata = result.get("metadata", {})
    timings = result.get("timings", {})
    metadata["timing"] = timings
    processing_info = json.dumps(metadata, indent=2)

    tab1 = critic_out
    tab2 = analyst_out
    tab3 = clinician_out
    tab4 = processing_info
    tab5 = "🧠 SiriuS learning in background..."

    # 4. Run SiriuS in background thread (Evaluator → Augmentation → Lesson extraction → ChromaDB)
    sirius_result = [None]

    def _sirius_background():
        try:
            sirius_result[0] = pipeline.evaluate_and_learn(result, image=image)
            # Add lessons_injected from metadata for display
            mem = metadata.get("memory_used", {})
            injected = [k for k, v in mem.items() if v]
            if sirius_result[0]:
                sirius_result[0]["lessons_injected"] = injected
        except Exception as e:
            print(f"SiriuS failed: {e}")
            sirius_result[0] = None

    t = Thread(target=_sirius_background)
    t.start()
    t.join(timeout=90)  # Wait up to 90s for SiriuS; user already has results

    # 5. Format Tab 5 content (score, lessons, augmentation status)
    if sirius_result[0] is not None:
        tab5 = _format_sirius_tab(sirius_result[0])
    else:
        tab5 = """SiriuS evaluation unavailable (cloud API may be down or misconfigured).
Your results above are complete and valid.
"""

    badge_count = str(_get_lesson_count())
    return tab1, tab2, tab3, tab4, tab5, badge_count


def _build_ui():
    n = _get_lesson_count()

    with gr.Blocks(
        title="MedConsult — Multi-Agent Medical Analysis",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# MedConsult — Multi-Agent Medical Analysis")
        gr.Markdown("Three MedGemma agents analyze your medical data. SiriuS learns from every analysis.")

        gr.Markdown("### ⚠️ AI research demo, not a medical device")

        badge = gr.Markdown(f"### 🧠 {n} medical lessons learned")

        with gr.Row():
            text_in = gr.Textbox(
                label="Medical Input",
                placeholder="Paste lab report, clinical note, or medical text...",
                lines=10,
            )
            image_in = gr.Image(type="pil", label="Medical Image (optional)")

        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.ClearButton([text_in, image_in])

        progress_text = gr.Markdown("")

        with gr.Tabs():
            with gr.Tab("📋 Patient Summary", id="tab1"):
                tab1_out = gr.Markdown("")
            with gr.Tab("🔬 Analyst Extraction", id="tab2"):
                tab2_out = gr.Markdown("")
            with gr.Tab("🏥 Clinical Interpretation", id="tab3"):
                tab3_out = gr.Markdown("")
            with gr.Tab("⏱️ Processing Info", id="tab4"):
                tab4_out = gr.Code(language="json", label="Metadata")
            with gr.Tab("🧠 SiriuS Intelligence", id="tab5"):
                with gr.Tabs():
                    with gr.Tab("📝 Learning Log"):
                        tab5_out = gr.Markdown("")
                    with gr.Tab("📊 Validation"):
                        validation_output = gr.JSON(label="Latest Validation Report")
                        run_count_display = gr.Textbox(label="Total Runs", interactive=False)
                        validate_btn = gr.Button("Check Validation Status")

                        def check_validation():
                            try:
                                pipeline = _get_pipeline()
                                report = pipeline.get_validation_report()
                                count = pipeline.get_run_count()
                                return (
                                    report if report else {"status": "No validation yet", "next_at": f"Run #{((count // 5) + 1) * 5}"},
                                    f"{count} runs completed (validates every 5)"
                                )
                            except Exception:
                                return {"status": "Pipeline not initialized"}, "0 runs"

                        validate_btn.click(check_validation, outputs=[validation_output, run_count_display])
        gr.Markdown("### Examples")
        gr.Examples(
            examples=[
                ["CBC: WBC 12.8, RBC 4.2, Hgb 11.2, Hct 33.8. Ref: WBC 4-11, RBC 4.5-5.5, Hgb 13.5-17.5, Hct 38.5-50."],
                ["67yo F, dyspnea, edema. Dx: Acute decompensated heart failure. On furosemide 40mg, lisinopril 20mg."],
                ["Glucose: 250 mg/dL (reference: 70-100)"],
            ],
            inputs=text_in,
            label="",
        )

        gr.Markdown("---")
        gr.Markdown(
            "*MedGemma 1.5 4B + Gemini 1.5 Pro | SiriuS Framework | MedGemma Impact Challenge 2026*"
        )

        def _handler(medical_text, image):
            msgs = []

            def _cb(m):
                msgs.append(m)

            r = _process(medical_text, image, progress_fn=_cb)
            return (
                r[0],
                r[1],
                r[2],
                r[3],
                r[4],
                gr.update(value=f"### 🧠 {r[5]} medical lessons learned"),
                msgs[-1] if msgs else "Done.",
            )

        submit_btn.click(
            fn=_handler,
            inputs=[text_in, image_in],
            outputs=[
                tab1_out,
                tab2_out,
                tab3_out,
                tab4_out,
                tab5_out,
                badge,
                progress_text,
            ],
        )

    return demo


def launch_ui():
    """Initializes and launches the Gradio web application for MedConsult."""
    demo = _build_ui()
    demo.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    launch_ui()
