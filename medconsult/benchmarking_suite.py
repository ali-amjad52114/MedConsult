"""
Comprehensive benchmarking and visualization suite for MedConsult pipeline.
Tracks performance, quality, and improvements over time with visual dashboards.
"""

import time
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

@dataclass
class BenchmarkMetrics:
    """Comprehensive metrics for each pipeline run."""
    timestamp: str
    input_type: str
    input_length: int
    has_image: bool
    
    # Performance metrics
    analyst_time: float
    clinician_time: float
    critic_time: float
    total_time: float
    memory_retrieval_time: float
    
    # Quality metrics
    sirius_score: int
    lessons_extracted: int
    lessons_injected: int
    augmentation_triggered: bool
    augmentation_attempts: int
    
    # Model metrics
    analyst_tokens: int
    clinician_tokens: int
    critic_tokens: int
    total_tokens: int
    
    # Memory metrics
    memory_used_analyst: bool
    memory_used_clinician: bool
    memory_used_critic: bool
    total_lessons_available: int
    
    # System metrics
    gpu_utilization: float
    memory_usage_mb: float
    model_load_time: float

class BenchmarkTracker:
    """Tracks and visualizes pipeline performance over time."""
    
    def __init__(self, results_dir: str = "benchmarking_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.metrics_file = self.results_dir / "benchmark_metrics.json"
        self.metrics: List[BenchmarkMetrics] = []
        self.load_existing_metrics()
        
    def load_existing_metrics(self):
        """Load previous benchmark results."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = [BenchmarkMetrics(**m) for m in data]
            except Exception as e:
                print(f"Warning: Could not load existing metrics: {e}")
    
    def save_metrics(self):
        """Save all metrics to JSON file."""
        data = [asdict(m) for m in self.metrics]
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_run(self, pipeline_result: Dict[str, Any], sirius_result: Dict[str, Any] = None):
        """Add a new pipeline run to metrics."""
        metadata = pipeline_result.get("metadata", {})
        timings = pipeline_result.get("timings", {})
        
        # Extract token counts (estimate from response lengths)
        analyst_tokens = len(pipeline_result.get("analyst", "").split()) * 1.3  # Rough estimate
        clinician_tokens = len(pipeline_result.get("clinician", "").split()) * 1.3
        critic_tokens = len(pipeline_result.get("critic", "").split()) * 1.3
        
        metric = BenchmarkMetrics(
            timestamp=metadata.get("timestamp", datetime.now(timezone.utc).isoformat()),
            input_type=metadata.get("input_type", "unknown"),
            input_length=len(pipeline_result.get("input", "")),
            has_image=False,  # TODO: Track image presence
            
            # Performance
            analyst_time=timings.get("analyst", 0),
            clinician_time=timings.get("clinician", 0),
            critic_time=timings.get("critic", 0),
            total_time=timings.get("total", 0),
            memory_retrieval_time=timings.get("memory_retrieval", 0),
            
            # Quality
            sirius_score=sirius_result.get("evaluation", {}).get("score", 0) if sirius_result else 0,
            lessons_extracted=sirius_result.get("lessons_extracted", 0) if sirius_result else 0,
            lessons_injected=sum(1 for v in (metadata.get("memory_used") or {}).values() if v),
            augmentation_triggered=sirius_result.get("augmented", False) if sirius_result else False,
            augmentation_attempts=sirius_result.get("augmentation_result", {}).get("attempts", 0) if sirius_result else 0,
            
            # Model metrics
            analyst_tokens=int(analyst_tokens),
            clinician_tokens=int(clinician_tokens),
            critic_tokens=int(critic_tokens),
            total_tokens=int(analyst_tokens + clinician_tokens + critic_tokens),
            
            # Memory
            memory_used_analyst=metadata.get("memory_used", {}).get("analyst", False),
            memory_used_clinician=metadata.get("memory_used", {}).get("clinician", False),
            memory_used_critic=metadata.get("memory_used", {}).get("critic", False),
            total_lessons_available=metadata.get("total_lessons_available", 0),
            
            # System (placeholders - implement with psutil for real values)
            gpu_utilization=0.0,
            memory_usage_mb=0.0,
            model_load_time=0.0,
        )
        
        self.metrics.append(metric)
        self.save_metrics()
        return metric
    
    def create_performance_dashboard(self, save_path: str = None):
        """Create comprehensive performance dashboard."""
        if not self.metrics:
            print("No metrics available for visualization")
            return
        
        df = pd.DataFrame([asdict(m) for m in self.metrics])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Response Times Over Time", "Quality Scores",
                "Token Generation", "Memory Usage",
                "Augmentation Impact", "Input Type Performance"
            ],
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # 1. Response Times
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['total_time'], name='Total Time', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['analyst_time'], name='Analyst', line=dict(color='red')),
            row=1, col=1
        )
        
        # 2. Quality Scores
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['sirius_score'], name='Sirius Score', 
                     line=dict(color='green'), marker=dict(size=8)),
            row=1, col=2
        )
        
        # 3. Token Generation
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['total_tokens'], name='Total Tokens',
                     line=dict(color='purple')),
            row=2, col=1
        )
        
        # 4. Memory Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['total_lessons_available'], name='Available Lessons',
                     line=dict(color='orange')),
            row=2, col=2
        )
        
        # 5. Augmentation Impact
        aug_counts = df['augmentation_triggered'].value_counts()
        fig.add_trace(
            go.Bar(x=aug_counts.index, y=aug_counts.values, name='Augmentation'),
            row=3, col=1
        )
        
        # 6. Input Type Performance
        input_perf = df.groupby('input_type')['total_time'].mean()
        fig.add_trace(
            go.Bar(x=input_perf.index, y=input_perf.values, name='Avg Time by Type'),
            row=3, col=2
        )
        
        fig.update_layout(
            height=1200, title_text="MedConsult Performance Dashboard",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def create_quality_trend_analysis(self, save_path: str = None):
        """Analyze quality improvement trends over time."""
        if len(self.metrics) < 5:
            print("Need at least 5 runs for trend analysis")
            return
        
        df = pd.DataFrame([asdict(m) for m in self.metrics])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate moving averages
        window = min(5, len(df) // 2)
        df['score_ma'] = df['sirius_score'].rolling(window=window).mean()
        df['lessons_ma'] = df['lessons_extracted'].rolling(window=window).mean()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Quality Score Trend", "Lesson Extraction Trend"],
            vertical_spacing=0.1
        )
        
        # Quality trend
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['sirius_score'], 
                     mode='markers', name='Individual Scores', 
                     marker=dict(color='lightblue', size=6)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['score_ma'], 
                     mode='lines', name=f'{window}-Run Moving Average',
                     line=dict(color='red', width=3)),
            row=1, col=1
        )
        
        # Lessons trend
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['lessons_extracted'], 
                     mode='markers', name='Lessons per Run',
                     marker=dict(color='lightgreen', size=6)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['lessons_ma'], 
                     mode='lines', name='Moving Average',
                     line=dict(color='darkgreen', width=3)),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800, title_text="Quality Improvement Trends Over Time"
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def generate_improvement_report(self) -> str:
        """Generate text report of improvements and recommendations."""
        if len(self.metrics) < 2:
            return "Need at least 2 runs for improvement analysis"
        
        recent = self.metrics[-10:]  # Last 10 runs
        early = self.metrics[:10]   # First 10 runs
        
        # Calculate improvements
        early_avg_score = sum(m.sirius_score for m in early) / len(early)
        recent_avg_score = sum(m.sirius_score for m in recent) / len(recent)
        score_improvement = recent_avg_score - early_avg_score
        
        early_avg_time = sum(m.total_time for m in early) / len(early)
        recent_avg_time = sum(m.total_time for m in recent) / len(recent)
        time_improvement = early_avg_time - recent_avg_time
        
        early_lessons = sum(m.lessons_extracted for m in early)
        recent_lessons = sum(m.lessons_extracted for m in recent)
        
        report = f"""
# MedConsult Improvement Report

## Performance Summary
- **Total Runs Analyzed**: {len(self.metrics)}
- **Time Period**: {self.metrics[0].timestamp[:10]} to {self.metrics[-1].timestamp[:10]}

## Quality Improvements
- **Early Average Score**: {early_avg_score:.2f}/5
- **Recent Average Score**: {recent_avg_score:.2f}/5
- **Improvement**: {score_improvement:+.2f} points ({(f'{score_improvement/early_avg_score*100:+.1f}%') if early_avg_score else 'N/A'})

## Performance Improvements  
- **Early Average Time**: {early_avg_time:.2f}s
- **Recent Average Time**: {recent_avg_time:.2f}s
- **Improvement**: {time_improvement:+.2f}s ({(f'{time_improvement/early_avg_time*100:+.1f}%') if early_avg_time else 'N/A'})

## Learning Effectiveness
- **Early Lessons Extracted**: {early_lessons} (avg {early_lessons/len(early):.1f}/run)
- **Recent Lessons Extracted**: {recent_lessons} (avg {recent_lessons/len(recent):.1f}/run)
- **Total Lessons in Memory**: {self.metrics[-1].total_lessons_available}

## Recommendations
"""
        
        # Add specific recommendations based on data
        if score_improvement < 0.5 and early_avg_score > 0:
            report += "- ⚠️ **Quality plateau detected**: Consider reviewing prompts or adding more diverse training data\n"
        if time_improvement < 0 and early_avg_time > 0:
            report += "- ⚠️ **Performance degradation**: Check for model loading issues or memory bottlenecks\n"
        if recent_lessons == 0:
            report += "- ⚠️ **No recent lessons**: SiriuS may need adjustment or quality threshold tuning\n"
        
        # Best performing input types
        input_performance = {}
        for m in recent:
            if m.input_type not in input_performance:
                input_performance[m.input_type] = []
            input_performance[m.input_type].append(m.sirius_score)
        
        if input_performance:
            best_type = max(input_performance.keys(),
                           key=lambda k: sum(input_performance[k]) / len(input_performance[k]))
            report += f"- 🏆 **Best performing input type**: {best_type} (avg score: {sum(input_performance[best_type])/len(input_performance[best_type]):.2f})\n"
        
        return report
    
    def export_detailed_csv(self, filename: str = None):
        """Export all metrics to CSV for detailed analysis."""
        if not filename:
            filename = f"medconsult_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame([asdict(m) for m in self.metrics])
        df.to_csv(filename, index=False)
        print(f"Metrics exported to {filename}")
        return filename


class RealTimeMonitor:
    """Real-time monitoring of pipeline execution."""
    
    def __init__(self, tracker: BenchmarkTracker):
        self.tracker = tracker
        self.current_run = None
        self.start_time = None
        
    def start_run(self, input_text: str, input_type: str):
        """Start monitoring a new pipeline run."""
        self.start_time = time.time()
        self.current_run = {
            "input_text": input_text,
            "input_type": input_type,
            "steps": [],
            "start_time": self.start_time
        }
        print(f"🚀 Starting pipeline run: {input_type}")
        
    def log_step(self, step_name: str, details: str = ""):
        """Log a step in the current run."""
        if self.current_run:
            elapsed = time.time() - self.start_time
            self.current_run["steps"].append({
                "step": step_name,
                "time": elapsed,
                "details": details
            })
            print(f"⏱️  [{elapsed:.2f}s] {step_name}: {details}")
    
    def end_run(self, pipeline_result: Dict, sirius_result: Dict = None):
        """End the current run and save metrics."""
        if self.current_run:
            total_time = time.time() - self.start_time
            self.current_run["total_time"] = total_time
            self.current_run["pipeline_result"] = pipeline_result
            self.current_run["sirius_result"] = sirius_result
            
            # Add to tracker
            metric = self.tracker.add_run(pipeline_result, sirius_result)
            
            print(f"✅ Run completed in {total_time:.2f}s")
            print(f"   Quality Score: {metric.sirius_score}/5")
            print(f"   Lessons Extracted: {metric.lessons_extracted}")
            print(f"   Total Lessons in Memory: {metric.total_lessons_available}")
            
            self.current_run = None


# Integration with existing pipeline
def create_benchmarked_pipeline(tracker: BenchmarkTracker, monitor: RealTimeMonitor):
    """Create a pipeline wrapper that automatically benchmarks everything."""
    from pipeline import MedConsultPipeline
    from sirius.memory_retriever import MemoryRetriever
    
    class BenchmarkedPipeline(MedConsultPipeline):
        def run_with_benchmarking(self, user_input_text: str, image=None):
            # Start monitoring
            input_type = self.experience_library.classify_input_type(user_input_text)
            monitor.start_run(user_input_text[:100] + "...", input_type)
            
            # Memory retrieval timing
            memory_start = time.time()
            retriever = MemoryRetriever(self.memory_store)
            analyst_memory = retriever.get_context_for_agent("analyst", user_input_text, input_type)
            clinician_memory = retriever.get_context_for_agent("clinician", user_input_text, input_type)
            critic_memory = retriever.get_context_for_agent("critic", user_input_text, input_type)
            memory_time = time.time() - memory_start
            monitor.log_step("Memory Retrieval", f"Retrieved lessons for 3 agents in {memory_time:.3f}s")
            
            # Analyst with timing
            analyst_start = time.time()
            analyst_output = self.analyst.analyze(user_input_text, image, memory_context=analyst_memory)
            analyst_time = time.time() - analyst_start
            monitor.log_step("Analyst", f"Completed in {analyst_time:.2f}s")
            
            # Clinician with timing
            clinician_start = time.time()
            clinician_output = self.clinician.interpret(user_input_text, analyst_output, image, memory_context=clinician_memory)
            clinician_time = time.time() - clinician_start
            monitor.log_step("Clinician", f"Completed in {clinician_time:.2f}s")
            
            # Critic with timing
            critic_start = time.time()
            critic_output = self.critic.review_and_communicate(user_input_text, analyst_output, clinician_output, image, memory_context=critic_memory)
            critic_time = time.time() - critic_start
            monitor.log_step("Critic", f"Completed in {critic_time:.2f}s")
            
            # Build result with enhanced timing
            result = {
                "input": user_input_text,
                "analyst": analyst_output,
                "clinician": clinician_output,
                "critic": critic_output,
                "timings": {
                    "memory_retrieval": memory_time,
                    "analyst": analyst_time,
                    "clinician": clinician_time,
                    "critic": critic_time,
                    "total": memory_time + analyst_time + clinician_time + critic_time
                },
                "metadata": {
                    "model": "google/medgemma-1.5-4b-it",
                    "meta_model": "gemini-2.0-flash",
                    "pipeline_version": "4.0-benchmarked",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "input_type": input_type,
                    "agent_chain": ["analyst", "clinician", "critic"],
                    "memory_used": {
                        "analyst": analyst_memory is not None,
                        "clinician": clinician_memory is not None,
                        "critic": critic_memory is not None,
                    },
                    "total_lessons_available": self.memory_store.get_lesson_count(),
                }
            }
            
            # End monitoring
            monitor.log_step("Pipeline Complete", f"Total time: {result['timings']['total']:.2f}s")
            
            return result
    
    return BenchmarkedPipeline()


if __name__ == "__main__":
    # Demo the benchmarking suite
    tracker = BenchmarkTracker()
    monitor = RealTimeMonitor(tracker)
    
    # Create benchmarked pipeline
    pipeline = create_benchmarked_pipeline(tracker, monitor)
    
    # Run some test cases
    test_cases = [
        "CBC: WBC 12.8, RBC 4.2, Hgb 11.2, Hct 33.8",
        "67yo F, dyspnea, edema. Dx: Acute decompensated heart failure.",
        "Glucose: 250 mg/dL (reference: 70-100)"
    ]
    
    print("🔬 Running benchmarked pipeline tests...")
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        result = pipeline.run_with_benchmarking(test_input)
        sirius_result = pipeline.evaluate_and_learn(result)
        monitor.end_run(result, sirius_result)
        
        time.sleep(1)  # Brief pause between runs
    
    # Generate visualizations
    print("\n📊 Generating performance dashboard...")
    tracker.create_performance_dashboard("benchmarking_results/dashboard.html")
    
    print("\n📈 Generating quality trend analysis...")
    tracker.create_quality_trend_analysis("benchmarking_results/trends.html")
    
    print("\n📋 Generating improvement report...")
    report = tracker.generate_improvement_report()
    with open("benchmarking_results/improvement_report.md", "w") as f:
        f.write(report)
    
    print("\n💾 Exporting detailed metrics...")
    tracker.export_detailed_csv("benchmarking_results/detailed_metrics.csv")
    
    print("\n✅ Benchmarking complete! Check benchmarking_results/ for all outputs.")
