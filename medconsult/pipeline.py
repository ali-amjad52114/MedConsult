"""
The main MedConsult pipeline: chains Analyst → Clinician → Critic,
with SiriuS self-improvement via evaluate_and_learn().
"""

import time
from datetime import datetime, timezone

from model.medgemma_manager import MedGemmaManager
from model.cloud_manager import CloudManager
from agents.analyst import AnalystAgent
from agents.clinician import ClinicianAgent
from agents.critic import CriticAgent
from agents.evaluator import EvaluatorAgent
from sirius.experience_library import ExperienceLibrary
from sirius.memory_store import MemoryStore
from sirius.lesson_extractor import LessonExtractor
from sirius.memory_retriever import MemoryRetriever
from sirius.augmentation import AugmentationLoop


class MedConsultPipeline:
    """Main pipeline: Analyst → Clinician → Critic with SiriuS evaluate_and_learn."""

    def __init__(self):
        # Model managers: MedGemma for clinical agents, Cloud (Gemini) for Evaluator + LessonExtractor
        medgemma = MedGemmaManager()
        cloud = CloudManager()

        # Clinical agents (all use MedGemma)
        self.analyst = AnalystAgent(medgemma)
        self.clinician = ClinicianAgent(medgemma)
        self.critic = CriticAgent(medgemma)
        # SiriuS components: Evaluator (Gemini), ExperienceLibrary (JSON chains), MemoryStore (ChromaDB)
        self.evaluator = EvaluatorAgent(cloud)
        self.experience_library = ExperienceLibrary()
        self.memory_store = MemoryStore()
        self.lesson_extractor = LessonExtractor(cloud)

    def run(self, user_input_text: str, image=None) -> dict:
        """
        Main user-facing function. Runs Analyst → Clinician → Critic with memory injection.
        """
        # Step 1: Classify input (lab_report, clinical_note, imaging, general)
        input_type = self.experience_library.classify_input_type(user_input_text)
        retriever = MemoryRetriever(self.memory_store)

        # Step 2: Retrieve relevant lessons from ChromaDB for each agent

        analyst_memory = retriever.get_context_for_agent("analyst", user_input_text, input_type)
        clinician_memory = retriever.get_context_for_agent("clinician", user_input_text, input_type)
        critic_memory = retriever.get_context_for_agent("critic", user_input_text, input_type)

        # Step 3: Run agent chain (each gets memory_context = learned lessons, or None)
        analyst_output = self.analyst.analyze(
            user_input_text, image, memory_context=analyst_memory
        )
        clinician_output = self.clinician.interpret(
            user_input_text, analyst_output, image, memory_context=clinician_memory
        )
        critic_output = self.critic.review_and_communicate(
            user_input_text, analyst_output, clinician_output, image,
            memory_context=critic_memory
        )

        return {
            "input": user_input_text,
            "analyst": analyst_output,
            "clinician": clinician_output,
            "critic": critic_output,
            "metadata": {
                "model": "google/medgemma-1.5-4b-it",
                "meta_model": "gemini-2.5-flash",
                "pipeline_version": "4.0-sirius-cloud",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_type": input_type,
                "agent_chain": ["analyst", "clinician", "critic"],
                "memory_used": {
                    "analyst": analyst_memory is not None,
                    "clinician": clinician_memory is not None,
                    "critic": critic_memory is not None,
                },
                "total_lessons_available": self.memory_store.get_lesson_count(),
            },
        }

    def evaluate_and_learn(self, pipeline_result: dict, image=None) -> dict:
        """
        SiriuS self-improvement: runs after user gets results.
        Flow: Evaluate → (Augment if score ≤ 2) → Save chain → Extract lessons → Store in ChromaDB
        """
        # Step 1: Gemini scores the chain 1–5, returns issues + improvements
        evaluation = self.evaluator.evaluate(
            pipeline_result["input"],
            pipeline_result["analyst"],
            pipeline_result["clinician"],
            pipeline_result["critic"],
        )

        # Step 2: If score ≤ 2, re-run agents with feedback (issues + improvements) injected
        if evaluation["score"] <= 2:
            loop = AugmentationLoop(self)
            improved = loop.augment(pipeline_result, evaluation, image)
            improved_chain = {
                **improved["improved_result"],
                "evaluation": improved["improved_evaluation"],
            }
            self.experience_library.save_chain(improved_chain, improved["final_score"])

            lessons = []
            if improved["final_score"] >= 3:
                lessons = self.lesson_extractor.extract(improved_chain)
                if lessons:
                    self.memory_store.store_lessons(lessons)

            return {
                "evaluation": evaluation,
                "augmented": True,
                "augmentation_result": improved,
                "lessons_extracted": len(lessons),
                "total_lessons": self.memory_store.get_lesson_count(),
                "library_stats": self.experience_library.get_stats(),
            }

        # Step 3: Save raw chain to good/ or bad/ JSON files
        chain_data = {**pipeline_result, "evaluation": evaluation}
        filepath = self.experience_library.save_chain(chain_data, evaluation["score"])

        # Step 4: If score ≥ 3, Gemini extracts lessons; store in ChromaDB for future runs
        lessons_stored = 0
        if evaluation["score"] >= 3:
            lessons = self.lesson_extractor.extract(chain_data)
            if lessons:
                lessons_stored = self.memory_store.store_lessons(lessons)

        return {
            "evaluation": evaluation,
            "saved_to": filepath,
            "lessons_extracted": lessons_stored,
            "total_lessons": self.memory_store.get_lesson_count(),
            "library_stats": self.experience_library.get_stats(),
        }

    def run_with_timing(self, user_input_text: str, image=None) -> dict:
        """Same as run() but includes per-agent timing in the result."""
        input_type = self.experience_library.classify_input_type(user_input_text)
        retriever = MemoryRetriever(self.memory_store)

        analyst_memory = retriever.get_context_for_agent("analyst", user_input_text, input_type)
        clinician_memory = retriever.get_context_for_agent("clinician", user_input_text, input_type)
        critic_memory = retriever.get_context_for_agent("critic", user_input_text, input_type)

        timings = {}

        t0 = time.time()
        analyst_output = self.analyst.analyze(
            user_input_text, image, memory_context=analyst_memory
        )
        timings["analyst"] = round(time.time() - t0, 2)

        t0 = time.time()
        clinician_output = self.clinician.interpret(
            user_input_text, analyst_output, image, memory_context=clinician_memory
        )
        timings["clinician"] = round(time.time() - t0, 2)

        t0 = time.time()
        critic_output = self.critic.review_and_communicate(
            user_input_text, analyst_output, clinician_output, image,
            memory_context=critic_memory
        )
        timings["critic"] = round(time.time() - t0, 2)

        timings["total"] = round(sum(timings.values()), 2)

        return {
            "input": user_input_text,
            "analyst": analyst_output,
            "clinician": clinician_output,
            "critic": critic_output,
            "timings": timings,
            "metadata": {
                "model": "google/medgemma-1.5-4b-it",
                "meta_model": "gemini-2.5-flash",
                "pipeline_version": "4.0-sirius-cloud",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_type": input_type,
                "agent_chain": ["analyst", "clinician", "critic"],
                "memory_used": {
                    "analyst": analyst_memory is not None,
                    "clinician": clinician_memory is not None,
                    "critic": critic_memory is not None,
                },
                "total_lessons_available": self.memory_store.get_lesson_count(),
            },
        }
