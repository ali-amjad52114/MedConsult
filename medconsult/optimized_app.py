"""
Optimized MedConsult app with faster pipeline integration.
"""

import time
import json
from threading import Thread

import gradio as gr


def _get_optimized_pipeline():
    """Lazy load optimized pipeline."""
    from optimized_pipeline import OptimizedMedConsultPipeline
    return OptimizedMedConsultPipeline(max_tokens=1024)  # Reduced tokens


def _process_optimized(
    medical_text: str,
    image,
    progress_fn,
) -> tuple[str, str, str, str, str, str]:
    """
    Optimized processing with faster pipeline.
    """
    text = (medical_text or "").strip()
    if not text and image is None:
        err = "Please provide medical text, an image, or both."
        return err, "", "", "{}", "SiriuS evaluation skipped (no input).", "0"

    if not text:
        text = "[Image only — no text provided]"

    try:
        pipeline = _get_optimized_pipeline()
    except Exception as e:
        return (
            f"Pipeline initialization failed: {e}",
            "", "", "{}", "SiriuS unavailable.", "0",
        )

    # Progress updates
    if progress_fn:
        progress_fn("🚀 Optimized: Retrieving memory in parallel...")
    
    if progress_fn:
        progress_fn("Step 1/3: Analyst extracting...")
    if progress_fn:
        progress_fn("Step 2/3: Clinician interpreting...")
    if progress_fn:
        progress_fn("Step 3/3: Critic reviewing...")

    # Run optimized pipeline
    result = pipeline.run_optimized(text, image=image)

    analyst_out = result.get("analyst", "")
    clinician_out = result.get("clinician", "")
    critic_out = result.get("critic", "")
    metadata = result.get("metadata", {})
    timings = result.get("timings", {})
    metadata["timing"] = timings
    metadata["optimization"] = "parallel_memory_retrieval+reduced_tokens"
    processing_info = json.dumps(metadata, indent=2)

    tab1 = critic_out
    tab2 = analyst_out
    tab3 = clinician_out
    tab4 = processing_info
    tab5 = "🧠 SiriuS learning in background..."

    # Run SiriuS in background (same as original)
    sirius_result = [None]

    def _sirius_background():
        try:
            sirius_result[0] = pipeline.evaluate_and_learn(result, image=image)
            mem = metadata.get("memory_used", {})
            injected = [k for k, v in mem.items() if v]
            if sirius_result[0]:
                sirius_result[0]["lessons_injected"] = injected
        except Exception as e:
            print(f"SiriuS failed: {e}")
            sirius_result[0] = None

    t = Thread(target=_sirius_background)
    t.start()
    t.join(timeout=60)  # Reduced timeout for faster feedback

    if sirius_result[0] is not None:
        from app import _format_sirius_tab
        tab5 = _format_sirius_tab(sirius_result[0])
    else:
        tab5 = """SiriuS evaluation unavailable (cloud API may be down).
Your results above are complete and valid.
"""

    badge_count = str(pipeline.memory_store.get_lesson_count())
    return tab1, tab2, tab3, tab4, tab5, badge_count


def _build_optimized_ui():
    """Build optimized UI with speed indicators."""
    pipeline = _get_optimized_pipeline()
    n = pipeline.memory_store.get_lesson_count()

    with gr.Blocks(
        title="MedConsult — Optimized",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# MedConsult — 🚀 Optimized Pipeline")
        gr.Markdown("Faster multi-agent medical analysis with parallel processing.")

        gr.Markdown("### ⚠️ AI research demo, not a medical device")

        badge = gr.Markdown(f"### 🧠 {n} medical lessons learned")

        with gr.Row():
            text_in = gr.Textbox(
                label="Medical Input",
                placeholder="Paste lab report, clinical note, or medical text...",
                lines=8,  # Reduced from 10
            )
            image_in = gr.Image(type="pil", label="Medical Image (optional)")

        with gr.Row():
            submit_btn = gr.Button("🚀 Submit (Optimized)", variant="primary")
            clear_btn = gr.ClearButton([text_in, image_in])

        progress_text = gr.Markdown("")

        with gr.Tabs():
            with gr.Tab("📋 Patient Summary", id="tab1"):
                tab1_out = gr.Markdown("")
            with gr.Tab("🔬 Analyst Extraction", id="tab2"):
                tab2_out = gr.Markdown("")
            with gr.Tab("🏥 Clinical Interpretation", id="tab3"):
                tab3_out = gr.Markdown("")
            with gr.Tab("⚡ Processing Info", id="tab4"):
                tab4_out = gr.Code(language="json", label="Metadata")
            with gr.Tab("🧠 SiriuS Intelligence", id="tab5"):
                tab5_out = gr.Markdown("")

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
            "*Optimized MedGemma 1.5 4B + Gemini 1.5 Pro | Parallel Processing | Reduced Tokens*"
        )

        def _handler(medical_text, image):
            msgs = []

            def _cb(m):
                msgs.append(m)

            r = _process_optimized(medical_text, image, progress_fn=_cb)
            return (
                r[0], r[1], r[2], r[3], r[4],
                gr.update(value=f"### 🧠 {r[5]} medical lessons learned"),
                msgs[-1] if msgs else "Done.",
            )

        submit_btn.click(
            fn=_handler,
            inputs=[text_in, image_in],
            outputs=[
                tab1_out, tab2_out, tab3_out, tab4_out, tab5_out, badge, progress_text,
            ],
        )

    return demo


def launch_optimized_ui():
    """Launch optimized Gradio interface."""
    demo = _build_optimized_ui()
    demo.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    launch_optimized_ui()
