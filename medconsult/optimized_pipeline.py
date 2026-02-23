"""
Optimized MedConsult pipeline with parallel execution for faster results.
"""

import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import threading

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


class OptimizedMedConsultPipeline:
    """Optimized pipeline with parallel memory retrieval and reduced tokens."""

    def __init__(self, max_tokens=1024):
        medgemma = MedGemmaManager()
        cloud = CloudManager()

        self.analyst = AnalystAgent(medgemma)
        self.clinician = ClinicianAgent(medgemma)
        self.critic = CriticAgent(medgemma)
        self.evaluator = EvaluatorAgent(cloud)
        self.experience_library = ExperienceLibrary()
        self.memory_store = MemoryStore()
        self.lesson_extractor = LessonExtractor(cloud)
        self.max_tokens = max_tokens  # Reduced for speed

    def _retrieve_memory_parallel(self, user_input_text: str, input_type: str):
        """Retrieve memory contexts for all agents in parallel."""
        retriever = MemoryRetriever(self.memory_store)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                'analyst': executor.submit(
                    retriever.get_context_for_agent, "analyst", user_input_text, input_type
                ),
                'clinician': executor.submit(
                    retriever.get_context_for_agent, "clinician", user_input_text, input_type
                ),
                'critic': executor.submit(
                    retriever.get_context_for_agent, "critic", user_input_text, input_type
                )
            }
            
            return {
                'analyst': futures['analyst'].result(),
                'clinician': futures['clinician'].result(), 
                'critic': futures['critic'].result()
            }

    def run_optimized(self, user_input_text: str, image=None) -> dict:
        """
        Optimized user-facing function with parallel memory retrieval.
        """
        input_type = self.experience_library.classify_input_type(user_input_text)
        
        # Parallel memory retrieval (faster than sequential)
        memory_contexts = self._retrieve_memory_parallel(user_input_text, input_type)

        # Sequential agent execution (required dependency chain)
        timings = {}
        
        t0 = time.time()
        analyst_output = self.analyst.analyze(
            user_input_text, image, memory_context=memory_contexts['analyst']
        )
        timings["analyst"] = round(time.time() - t0, 2)

        t0 = time.time()
        clinician_output = self.clinician.interpret(
            user_input_text, analyst_output, image, memory_context=memory_contexts['clinician']
        )
        timings["clinician"] = round(time.time() - t0, 2)

        t0 = time.time()
        critic_output = self.critic.review_and_communicate(
            user_input_text, analyst_output, clinician_output, image,
            memory_context=memory_contexts['critic']
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
                "pipeline_version": "4.1-optimized",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_type": input_type,
                "agent_chain": ["analyst", "clinician", "critic"],
                "memory_used": {
                    "analyst": memory_contexts['analyst'] is not None,
                    "clinician": memory_contexts['clinician'] is not None,
                    "critic": memory_contexts['critic'] is not None,
                },
                "total_lessons_available": self.memory_store.get_lesson_count(),
                "optimization": "parallel_memory_retrieval"
            },
        }

    def evaluate_and_learn(self, pipeline_result: dict, image=None) -> dict:
        """Same as original - runs in background."""
        # Step 1: Evaluate with cloud model
        evaluation = self.evaluator.evaluate(
            pipeline_result["input"],
            pipeline_result["analyst"],
            pipeline_result["clinician"],
            pipeline_result["critic"],
        )

        # Step 2: Low score → augmentation loop
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

        # Step 3: Save raw chain
        chain_data = {**pipeline_result, "evaluation": evaluation}
        filepath = self.experience_library.save_chain(chain_data, evaluation["score"])

        # Step 4: Extract and store lessons from good chains
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
