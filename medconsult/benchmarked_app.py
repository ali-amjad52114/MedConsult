"""
MedConsult app with comprehensive benchmarking and visualization integration.
Shows real-time monitoring, performance tracking, and improvement analysis.
"""

import time
import json
from threading import Thread
import gradio as gr
from benchmarking_suite import BenchmarkTracker, RealTimeMonitor, create_benchmarked_pipeline
from visual_dashboard import MonitoredPipeline, PipelineVisualizer

# Global tracking objects
benchmark_tracker = BenchmarkTracker()
realtime_monitor = RealTimeMonitor(benchmark_tracker)
pipeline_visualizer = PipelineVisualizer()

def _get_benchmarked_pipeline():
    """Lazy load benchmarked pipeline."""
    return create_benchmarked_pipeline(benchmark_tracker, realtime_monitor)

def _process_with_monitoring(
    medical_text: str,
    image,
    progress_fn,
) -> tuple[str, str, str, str, str, str]:
    """
    Processing with comprehensive benchmarking and real-time monitoring.
    """
    text = (medical_text or "").strip()
    if not text and image is None:
        err = "Please provide medical text, an image, or both."
        return err, "", "", "{}", "📊 Benchmarking skipped (no input).", "0"

    if not text:
        text = "[Image only — no text provided]"

    try:
        pipeline = _get_benchmarked_pipeline()
    except Exception as e:
        return (
            f"Pipeline initialization failed: {e}",
            "", "", "{}", "📊 Benchmarking unavailable.", "0",
        )

    # Progress updates with monitoring
    if progress_fn:
        progress_fn("🔍 Starting comprehensive monitoring...")
    
    # Run benchmarked pipeline
    result = pipeline.run_with_benchmarking(text, image=image)

    analyst_out = result.get("analyst", "")
    clinician_out = result.get("clinician", "")
    critic_out = result.get("critic", "")
    metadata = result.get("metadata", {})
    timings = result.get("timings", {})
    metadata["timing"] = timings
    metadata["benchmarking_enabled"] = True
    processing_info = json.dumps(metadata, indent=2)

    tab1 = critic_out
    tab2 = analyst_out
    tab3 = clinician_out
    tab4 = processing_info
    tab5 = "📊 SiriuS learning with benchmarking in background..."

    # Run SiriuS with benchmarking
    sirius_result = [None]

    def _sirius_background():
        try:
            # Log SiriuS start
            pipeline_visualizer.log_step("SiriuS Evaluation", "Evaluator", "starting")
            
            sirius_result[0] = pipeline.evaluate_and_learn(result, image=image)
            
            # Log completion
            pipeline_visualizer.log_step(
                "SiriuS Complete", "Evaluator", "completed",
                metadata={
                    "score": sirius_result[0].get("evaluation", {}).get("score", 0),
                    "lessons_extracted": sirius_result[0].get("lessons_extracted", 0),
                    "augmented": sirius_result[0].get("augmented", False)
                }
            )
            
            # Add lesson count badge update
            mem = metadata.get("memory_used", {})
            injected = [k for k, v in mem.items() if v]
            if sirius_result[0]:
                sirius_result[0]["lessons_injected"] = injected
                
        except Exception as e:
            pipeline_visualizer.log_step("SiriuS Error", "Evaluator", "error", metadata={"error": str(e)})
            print(f"SiriuS failed: {e}")
            sirius_result[0] = None

    t = Thread(target=_sirius_background)
    t.start()
    t.join(timeout=90)

    # Record metrics for benchmarking (must run after SiriuS completes)
    realtime_monitor.end_run(result, sirius_result[0])

    if sirius_result[0] is not None:
        from app import _format_sirius_tab
        tab5 = _format_sirius_tab(sirius_result[0])
    else:
        tab5 = """📊 SiriuS evaluation unavailable (cloud API may be down).
Your results above are complete and valid.
"""

    if sirius_result[0] is not None:
        badge_count = str(sirius_result[0].get("total_lessons", 0))
    elif benchmark_tracker.metrics:
        badge_count = str(benchmark_tracker.metrics[-1].total_lessons_available)
    else:
        badge_count = "0"
    return tab1, tab2, tab3, tab4, tab5, badge_count

def _build_benchmarked_ui():
    """Build UI with benchmarking controls and visualizations."""
    n = benchmark_tracker.metrics[-1].total_lessons_available if benchmark_tracker.metrics else 0

    with gr.Blocks(
        title="MedConsult — Benchmarking Edition",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# MedConsult — 📊 Benchmarking & Monitoring Edition")
        gr.Markdown("Real-time monitoring of every LLM step with comprehensive performance tracking.")

        gr.Markdown("### ⚠️ AI research demo, not a medical device")

        # Metrics header (updateable badges)
        with gr.Row():
            badge_lessons = gr.Markdown(f"### 🧠 {n} medical lessons learned")
            badge_runs = gr.Markdown(f"### 📈 {len(benchmark_tracker.metrics)} runs tracked")
            badge_avg = gr.Markdown(f"### ⭐ Avg Quality: {(sum(m.sirius_score for m in benchmark_tracker.metrics) / len(benchmark_tracker.metrics)):.2f}/5" if benchmark_tracker.metrics else "### ⭐ Avg Quality: —")

        with gr.Row():
            text_in = gr.Textbox(
                label="Medical Input",
                placeholder="Paste lab report, clinical note, or medical text...",
                lines=8,
            )
            image_in = gr.Image(type="pil", label="Medical Image (optional)")

        with gr.Row():
            submit_btn = gr.Button("📊 Submit (With Benchmarking)", variant="primary")
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
            with gr.Tab("📊 SiriuS Intelligence", id="tab5"):
                tab5_out = gr.Markdown("")

        # Benchmarking controls
        gr.Markdown("---")
        gr.Markdown("### 📈 Benchmarking Controls")
        
        with gr.Row():
            generate_dashboard_btn = gr.Button("📊 Generate Performance Dashboard")
            generate_trends_btn = gr.Button("📈 Generate Quality Trends")
            export_metrics_btn = gr.Button("💾 Export Metrics")
            view_report_btn = gr.Button("📋 View Improvement Report")

        # Benchmarking outputs
        with gr.Row():
            dashboard_output = gr.File(label="Performance Dashboard (HTML)")
            trends_output = gr.File(label="Quality Trends (HTML)")
            metrics_output = gr.File(label="Detailed Metrics (CSV)")
            report_output = gr.Markdown("")

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
            "*Benchmarked MedGemma 1.5 4B + Gemini 1.5 Pro | Real-time Monitoring | Performance Analytics*"
        )

        def _handler(medical_text, image):
            msgs = []

            def _cb(m):
                msgs.append(m)

            r = _process_with_monitoring(medical_text, image, progress_fn=_cb)
            avg = (sum(m.sirius_score for m in benchmark_tracker.metrics) / len(benchmark_tracker.metrics)) if benchmark_tracker.metrics else 0
            return (
                r[0], r[1], r[2], r[3], r[4],
                gr.update(value=f"### 🧠 {r[5]} medical lessons learned"),
                gr.update(value=f"### 📈 {len(benchmark_tracker.metrics)} runs tracked"),
                gr.update(value=f"### ⭐ Avg Quality: {avg:.2f}/5" if benchmark_tracker.metrics else "### ⭐ Avg Quality: —"),
                msgs[-1] if msgs else "Done.",
            )

        def _generate_dashboard():
            """Generate performance dashboard."""
            dashboard_path = "benchmarking_results/dashboard.html"
            benchmark_tracker.create_performance_dashboard(dashboard_path)
            return dashboard_path

        def _generate_trends():
            """Generate quality trend analysis."""
            trends_path = "benchmarking_results/trends.html"
            benchmark_tracker.create_quality_trend_analysis(trends_path)
            return trends_path

        def _export_metrics():
            """Export detailed metrics."""
            csv_path = benchmark_tracker.export_detailed_csv()
            return csv_path

        def _view_report():
            """Generate improvement report."""
            report = benchmark_tracker.generate_improvement_report()
            return report

        # Event handlers
        submit_btn.click(
            fn=_handler,
            inputs=[text_in, image_in],
            outputs=[
                tab1_out, tab2_out, tab3_out, tab4_out, tab5_out,
                badge_lessons, badge_runs, badge_avg, progress_text
            ],
        )

        generate_dashboard_btn.click(
            fn=_generate_dashboard,
            outputs=[dashboard_output]
        )

        generate_trends_btn.click(
            fn=_generate_trends,
            outputs=[trends_output]
        )

        export_metrics_btn.click(
            fn=_export_metrics,
            outputs=[metrics_output]
        )

        view_report_btn.click(
            fn=_view_report,
            outputs=[report_output]
        )

    return demo


def launch_benchmarked_ui():
    """Launch benchmarked Gradio interface."""
    demo = _build_benchmarked_ui()
    demo.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    launch_benchmarked_ui()
