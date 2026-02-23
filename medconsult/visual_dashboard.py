"""
Real-time visual dashboard for MedConsult pipeline monitoring.
Shows what each LLM is doing, step-by-step progress, and performance metrics.
"""

import streamlit as st
import time
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue

class PipelineVisualizer:
    """Real-time visualization of pipeline execution."""
    
    def __init__(self):
        self.step_queue = queue.Queue()
        self.metrics_history = []
        self.current_run = None
        
    def log_step(self, step_name: str, agent: str, status: str, 
                 input_text: str = "", output_text: str = "", 
                 timing: float = 0, metadata: dict = None):
        """Log a pipeline step for visualization."""
        step_data = {
            "timestamp": datetime.now(),
            "step_name": step_name,
            "agent": agent,
            "status": status,  # "starting", "processing", "completed", "error"
            "input_preview": input_text[:200] + "..." if len(input_text) > 200 else input_text,
            "output_preview": output_text[:200] + "..." if len(output_text) > 200 else output_text,
            "timing": timing,
            "metadata": metadata or {}
        }
        self.step_queue.put(step_data)
    
    def get_recent_steps(self, limit: int = 50):
        """Get recent pipeline steps."""
        steps = []
        temp_queue = queue.Queue()
        
        # Extract all items from queue
        while not self.step_queue.empty():
            try:
                step = self.step_queue.get_nowait()
                steps.append(step)
                temp_queue.put(step)
            except queue.Empty:
                break
        
        # Put items back and add new ones
        while not temp_queue.empty():
            self.step_queue.put(temp_queue.get())
        
        return steps[-limit:] if steps else []
    
    def create_agent_flow_diagram(self):
        """Create visual flow diagram of agent interactions."""
        fig = go.Figure()
        
        # Define agent positions
        agents = {
            "Input": (1, 3),
            "Memory": (2, 3),
            "Analyst": (3, 3),
            "Clinician": (4, 3),
            "Critic": (5, 3),
            "Evaluator": (6, 2),
            "Lesson Extractor": (6, 1),
            "User": (5, 1)
        }
        
        # Add nodes
        for agent, (x, y) in agents.items():
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=x-0.3, y0=y-0.3, x1=x+0.3, y1=y+0.3,
                fillcolor="lightblue" if agent != "User" else "lightgreen",
                line=dict(color="navy", width=2)
            )
            fig.add_annotation(
                x=x, y=y, text=agent,
                showarrow=False, font=dict(size=10, color="navy")
            )
        
        # Add flow arrows
        flows = [
            ("Input", "Memory"), ("Memory", "Analyst"), 
            ("Analyst", "Clinician"), ("Clinician", "Critic"),
            ("Critic", "User"), ("Critic", "Evaluator"),
            ("Evaluator", "Lesson Extractor"), ("Lesson Extractor", "Memory")
        ]
        
        for from_agent, to_agent in flows:
            x1, y1 = agents[from_agent]
            x2, y2 = agents[to_agent]
            
            fig.add_annotation(
                x=x1, y=y1, ax=x2-x1, ay=y2-y1,
                arrowhead=2, arrowsize=1, arrowwidth=2,
                arrowcolor="gray"
            )
        
        fig.update_layout(
            title="MedConsult Agent Flow Diagram",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400, showlegend=False
        )
        
        return fig
    
    def create_real_time_timeline(self):
        """Create real-time timeline of pipeline steps."""
        steps = self.get_recent_steps()
        if not steps:
            return go.Figure()
        
        df = pd.DataFrame(steps)
        df["Start"] = pd.to_datetime(df["timestamp"])
        df["Finish"] = df["Start"] + pd.to_timedelta(df["timing"].fillna(0), unit="s")

        # Create timeline (px.timeline expects Start/Finish columns)
        fig = px.timeline(
            df, x_start="Start", x_end="Finish", y="agent",
            color="status", title="Pipeline Execution Timeline",
            color_discrete_map={
                "starting": "yellow",
                "processing": "orange", 
                "completed": "green",
                "error": "red"
            }
        )
        
        fig.update_layout(height=300)
        return fig
    
    def create_performance_metrics(self):
        """Create performance metrics dashboard."""
        steps = self.get_recent_steps()
        if not steps:
            return go.Figure()
        
        # Calculate metrics by agent
        agent_metrics = {}
        for step in steps:
            agent = step["agent"]
            if agent not in agent_metrics:
                agent_metrics[agent] = []
            agent_metrics[agent].append(step["timing"])
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Agent Response Times", "Step Status Distribution", 
                         "Timeline View", "Performance Trend"]
        )
        
        # 1. Response times by agent
        agents = list(agent_metrics.keys())
        avg_times = [sum(agent_metrics[a])/len(agent_metrics[a]) for a in agents]
        
        fig.add_trace(
            go.Bar(x=agents, y=avg_times, name="Avg Time"),
            row=1, col=1
        )
        
        # 2. Status distribution
        status_counts = {}
        for step in steps:
            status = step["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        fig.add_trace(
            go.Pie(labels=list(status_counts.keys()), 
                   values=list(status_counts.values()),
                   name="Status Distribution"),
            row=1, col=2
        )
        
        # 3. Timeline scatter
        timestamps = [step["timestamp"] for step in steps]
        timings = [step["timing"] for step in steps]
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=timings, mode='markers+lines',
                     name="Step Timing"),
            row=2, col=1
        )
        
        # 4. Performance trend (last 20 steps)
        recent_steps = steps[-20:]
        if len(recent_steps) > 1:
            trend_times = [step["timing"] for step in recent_steps]
            fig.add_trace(
                go.Scatter(y=trend_times, mode='lines',
                         name="Recent Trend"),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=False)
        return fig


def create_streamlit_dashboard():
    """Create Streamlit dashboard for real-time monitoring."""
    visualizer = PipelineVisualizer()
    
    st.title("🏥 MedConsult Pipeline Real-Time Monitor")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (s)", 1, 10, 2)
    
    # Main dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Agent Flow Diagram")
        fig_flow = visualizer.create_agent_flow_diagram()
        st.plotly_chart(fig_flow, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Real-Time Timeline")
        fig_timeline = visualizer.create_real_time_timeline()
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    st.markdown("---")
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    fig_metrics = visualizer.create_performance_metrics()
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Recent activity log
    st.subheader("📝 Recent Activity Log")
    steps = visualizer.get_recent_steps(10)
    
    if steps:
        for i, step in enumerate(reversed(steps[-5:]), 1):
            status_emoji = {
                "starting": "🟡", "processing": "🟠", 
                "completed": "🟢", "error": "🔴"
            }.get(step["status"], "⚪")
            
            with st.expander(f"{status_emoji} {step['step_name']} - {step['agent']} ({step['timestamp'].strftime('%H:%M:%S')})"):
                st.write(f"**Status:** {step['status']}")
                st.write(f"**Timing:** {step['timing']:.3f}s")
                if step['input_preview']:
                    st.write(f"**Input:** {step['input_preview']}")
                if step['output_preview']:
                    st.write(f"**Output:** {step['output_preview']}")
                if step['metadata']:
                    st.write(f"**Metadata:** {step['metadata']}")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


def create_llm_transparency_panel():
    """Create detailed panel showing what each LLM is doing."""
    st.title("🔍 LLM Transparency Panel")
    
    # Agent details section
    st.header("Agent-by-Agent Analysis")
    
    agents_info = {
        "Analyst": {
            "model": "MedGemma 1.5 4B",
            "purpose": "Extract factual medical data",
            "input": "Raw medical text + images",
            "output": "Structured findings",
            "tokens": "~1024",
            "time": "~2-4s"
        },
        "Clinician": {
            "model": "MedGemma 1.5 4B", 
            "purpose": "Interpret patterns and differentials",
            "input": "Original + Analyst output",
            "output": "Clinical interpretation",
            "tokens": "~1024",
            "time": "~2-4s"
        },
        "Critic": {
            "model": "MedGemma 1.5 4B",
            "purpose": "Patient-friendly summary",
            "input": "Original + Analyst + Clinician",
            "output": "Plain language summary",
            "tokens": "~2048",
            "time": "~3-5s"
        },
        "Evaluator": {
            "model": "Gemini 1.5 Pro",
            "purpose": "Quality scoring 1-5",
            "input": "Complete analysis chain",
            "output": "Score + feedback",
            "tokens": "~512",
            "time": "~1-2s"
        },
        "Lesson Extractor": {
            "model": "Gemini 1.5 Pro",
            "purpose": "Distill reasoning patterns",
            "input": "High-quality chains",
            "output": "Structured lessons",
            "tokens": "~1024",
            "time": "~2-3s"
        }
    }
    
    for agent_name, info in agents_info.items():
        with st.expander(f"🤖 {agent_name} Agent"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Model:** {info['model']}")
                st.write(f"**Purpose:** {info['purpose']}")
                st.write(f"**Tokens:** {info['tokens']}")
                
            with col2:
                st.write(f"**Input:** {info['input']}")
                st.write(f"**Output:** {info['output']}")
                st.write(f"**Time:** {info['time']}")
    
    # System metrics
    st.header("📈 System Performance Metrics")
    
    # Placeholder for real metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", "47", "↑ 12 this week")
    
    with col2:
        st.metric("Avg Quality Score", "4.2/5", "↑ 0.3 from baseline")
    
    with col3:
        st.metric("Avg Response Time", "8.3s", "↓ 1.2s from baseline")
    
    with col4:
        st.metric("Lessons Learned", "156", "↑ 24 this week")
    
    # Quality improvement graph
    st.subheader("📈 Quality Improvement Over Time")
    
    # Sample data for demonstration
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    scores = [3.2 + i*0.03 + (i%3)*0.1 for i in range(30)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=scores, mode='lines+markers', name='Quality Score'))
    fig.add_trace(go.Scatter(x=dates, y=pd.Series(scores).rolling(7).mean(), 
                         mode='lines', name='7-Day Moving Avg', line=dict(width=3)))
    
    fig.update_layout(
        title="SiriuS Quality Score Trend",
        xaxis_title="Date",
        yaxis_title="Quality Score (1-5)",
        yaxis=dict(range=[1, 5])
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="MedConsult Pipeline Monitor",
        page_icon="🏥",
        layout="wide",
    )
    # Choose dashboard type
    dashboard_type = st.sidebar.selectbox(
        "Dashboard Type",
        ["Real-Time Monitor", "LLM Transparency", "Agent Analysis"]
    )
    
    if dashboard_type == "Real-Time Monitor":
        create_streamlit_dashboard()
    elif dashboard_type == "LLM Transparency":
        create_llm_transparency_panel()
    else:
        st.title("🔬 Agent Analysis Dashboard")
        st.write("Detailed agent analysis coming soon...")


# Integration with existing pipeline
class MonitoredPipeline:
    """Pipeline wrapper with real-time monitoring."""

    def __init__(self, visualizer: PipelineVisualizer):
        self.visualizer = visualizer
        from pipeline import MedConsultPipeline
        self.pipeline = MedConsultPipeline()
    
    def run_with_monitoring(self, user_input_text: str, image=None):
        """Run pipeline with real-time monitoring."""
        
        # Step 1: Memory Retrieval
        self.visualizer.log_step(
            "Memory Retrieval", "Memory", "starting",
            input_text=user_input_text[:100]
        )
        
        memory_start = time.time()
        input_type = self.pipeline.experience_library.classify_input_type(user_input_text)
        from sirius.memory_retriever import MemoryRetriever
        retriever = MemoryRetriever(self.pipeline.memory_store)
        
        analyst_memory = retriever.get_context_for_agent("analyst", user_input_text, input_type)
        clinician_memory = retriever.get_context_for_agent("clinician", user_input_text, input_type)
        critic_memory = retriever.get_context_for_agent("critic", user_input_text, input_type)
        
        memory_time = time.time() - memory_start
        lessons_found = sum(1 for m in [analyst_memory, clinician_memory, critic_memory] if m is not None)
        self.visualizer.log_step(
            "Memory Retrieval", "Memory", "completed",
            timing=memory_time,
            metadata={"lessons_found": lessons_found}
        )
        
        # Step 2: Analyst
        self.visualizer.log_step("Medical Analysis", "Analyst", "starting")
        analyst_start = time.time()
        analyst_output = self.pipeline.analyst.analyze(user_input_text, image, memory_context=analyst_memory)
        analyst_time = time.time() - analyst_start
        self.visualizer.log_step(
            "Medical Analysis", "Analyst", "completed",
            output_text=analyst_output[:200],
            timing=analyst_time
        )
        
        # Step 3: Clinician
        self.visualizer.log_step("Clinical Interpretation", "Clinician", "starting")
        clinician_start = time.time()
        clinician_output = self.pipeline.clinician.interpret(user_input_text, analyst_output, image, memory_context=clinician_memory)
        clinician_time = time.time() - clinician_start
        self.visualizer.log_step(
            "Clinical Interpretation", "Clinician", "completed",
            output_text=clinician_output[:200],
            timing=clinician_time
        )
        
        # Step 4: Critic
        self.visualizer.log_step("Patient Summary", "Critic", "starting")
        critic_start = time.time()
        critic_output = self.pipeline.critic.review_and_communicate(user_input_text, analyst_output, clinician_output, image, memory_context=critic_memory)
        critic_time = time.time() - critic_start
        self.visualizer.log_step(
            "Patient Summary", "Critic", "completed",
            output_text=critic_output[:200],
            timing=critic_time
        )
        
        # Return results
        return {
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
                "input_type": input_type,
                "memory_used": {
                    "analyst": analyst_memory is not None,
                    "clinician": clinician_memory is not None,
                    "critic": critic_memory is not None,
                }
            }
        }
