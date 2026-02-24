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
from sirius.augmentation import EnhancedAugmentation
from sirius.validator import PeriodicValidator


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
        self.run_count = 0
        self.chain_history = []
        self.validator = PeriodicValidator(cloud, validation_interval=5)
        self.augmenter = EnhancedAugmentation(cloud, max_retries=3)

    def run(self, user_input_text: str, image=None) -> dict:
        """
        Main user-facing function. Runs Analyst → Clinician → Critic with memory injection.
        """
        # Step 1: Classify input (lab_report, clinical_note, imaging, general)
        input_type = self.experience_library.classify_input_type(user_input_text)
        retriever = MemoryRetriever(self.memory_store)

        # Step 2: Retrieve relevant lessons from ChromaDB for each agent

        analyst_memory = retriever.get_relevant_lessons(user_input_text, agent_name="analyst")
        clinician_memory = retriever.get_relevant_lessons(user_input_text, agent_name="clinician")
        critic_memory = retriever.get_relevant_lessons(user_input_text, agent_name="critic")

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

        self.run_count += 1
        
        # Store chain for validation
        self.chain_history.append({
            "name": pipeline_result.get("metadata", {}).get("input_type", "unknown"),
            "input": pipeline_result["input"],
            "analyst": pipeline_result["analyst"],
            "clinician": pipeline_result["clinician"],
            "critic": pipeline_result["critic"],
            "score": evaluation.get("score", 0),
        })

        if len(self.chain_history) > 10:
            self.chain_history = self.chain_history[-10:]

        validation_report = None
        if self.validator.should_validate(self.run_count):
            print(f"INFO: Triggering periodic validation (run #{self.run_count})")
            validation_report = self.validator.run(self.chain_history[-5:])

        # Step 2: If score ≤ 2, re-run agents with feedback (issues + improvements) injected
        if evaluation["score"] <= 2:
            print(f"INFO: Score {evaluation['score']}/5 — triggering enhanced augmentation")

            chain = {
                "input": pipeline_result["input"],
                "analyst": pipeline_result["analyst"],
                "clinician": pipeline_result["clinician"],
                "critic": pipeline_result["critic"],
            }

            # (a) Per-agent scoring
            agent_scores = self.augmenter.score_agents(chain)
            retry_list = self.augmenter.get_retry_agents(agent_scores)

            score = evaluation["score"]
            analyst_output = pipeline_result["analyst"]
            clinician_output = pipeline_result["clinician"]
            critic_output = pipeline_result["critic"]
            new_eval = evaluation

            # (b) Targeted escalating retry
            for agent_name, feedback in retry_list:
                for attempt in range(1, self.augmenter.max_retries + 1):
                    context, strategy = self.augmenter.get_retry_context(
                        agent_name, feedback, attempt
                    )
                    print(f"  Retry {attempt}: {agent_name} using {strategy}")

                    if agent_name == "analyst":
                        analyst_output = self.analyst.analyze(
                            pipeline_result["input"], image, memory_context=context
                        )
                    elif agent_name == "clinician":
                        clinician_output = self.clinician.interpret(
                            pipeline_result["input"], analyst_output, image, memory_context=context
                        )
                    elif agent_name == "critic":
                        critic_output = self.critic.review_and_communicate(
                            pipeline_result["input"], analyst_output, clinician_output, image,
                            memory_context=context
                        )

                    # Re-evaluate after retry
                    new_eval = self.evaluator.evaluate(
                        pipeline_result["input"], analyst_output, clinician_output, critic_output
                    )
                    new_score = new_eval.get("score", 0)
                    print(f"  New score: {new_score}/5")

                    if new_score >= 3:
                        print(f"  ✓ {agent_name} improved to {new_score}")
                        score = new_score
                        break

            # (c) Cross-agent feedback (store for next run)
            cross_fb = self.augmenter.get_cross_feedback(critic_output)
            if cross_fb:
                print(f"  Cross-feedback: {cross_fb[:100]}")

            # (d) Anti-lessons from failure
            anti_lessons = self.augmenter.extract_anti_lessons(chain, score)
            for al in anti_lessons:
                self.memory_store.add_lesson(
                    al["rule"],
                    metadata={
                        "target_agent": al["target_agent"],
                        "lesson_type": "pitfall_warning",
                        "topic": al["topic"],
                        "confidence": "high",
                        "source": "anti_lesson",
                    }
                )
            if anti_lessons:
                print(f"  Stored {len(anti_lessons)} anti-lessons")
                
            improved_chain = {
                "input": pipeline_result["input"],
                "analyst": analyst_output,
                "clinician": clinician_output,
                "critic": critic_output,
                "metadata": pipeline_result.get("metadata", {}),
                "evaluation": new_eval if score >= 3 else evaluation
            }
            self.experience_library.save_chain(improved_chain, score)
                
            lessons_extracted = 0
            if score >= 3:
                lessons = self.lesson_extractor.extract(improved_chain)
                for lesson in lessons:
                    self.memory_store.add_lesson(
                        lesson.get("rule", ""),
                        metadata={
                            "target_agent": lesson.get("target_agent", ""),
                            "lesson_type": lesson.get("lesson_type", ""),
                            "topic": lesson.get("topic", ""),
                            "confidence": lesson.get("confidence", ""),
                            "source": "sirius_extractor",
                        }
                    )
                    lessons_extracted += 1

            return {
                "evaluation": evaluation,
                "augmented": True,
                "augmentation_result": {"final_score": score},
                "lessons_extracted": lessons_extracted,
                "total_lessons": self.memory_store.get_lesson_count(),
                "library_stats": self.experience_library.get_stats(),
                "validation_report": validation_report,
            }

        # Step 3: Save raw chain to good/ or bad/ JSON files
        chain_data = {**pipeline_result, "evaluation": evaluation}
        filepath = self.experience_library.save_chain(chain_data, evaluation["score"])

        # Step 4: If score ≥ 3, Gemini extracts lessons; store in ChromaDB for future runs.
        lessons_stored = 0
        if evaluation["score"] >= 3:
            lessons = self.lesson_extractor.extract(chain_data)
            for lesson in lessons:
                self.memory_store.add_lesson(
                    lesson.get("rule", ""),
                    metadata={
                        "target_agent": lesson.get("target_agent", ""),
                        "lesson_type": lesson.get("lesson_type", ""),
                        "topic": lesson.get("topic", ""),
                        "confidence": lesson.get("confidence", ""),
                        "source": "sirius_extractor",
                    }
                )
                lessons_stored += 1

        return {
            "evaluation": evaluation,
            "saved_to": filepath,
            "lessons_extracted": lessons_stored,
            "total_lessons": self.memory_store.get_lesson_count(),
            "library_stats": self.experience_library.get_stats(),
            "validation_report": validation_report,
        }

    def run_with_timing(self, user_input_text: str, image=None) -> dict:
        """Same as run() but includes per-agent timing in the result."""
        input_type = self.experience_library.classify_input_type(user_input_text)
        retriever = MemoryRetriever(self.memory_store)

        analyst_memory = retriever.get_relevant_lessons(user_input_text, agent_name="analyst")
        clinician_memory = retriever.get_relevant_lessons(user_input_text, agent_name="clinician")
        critic_memory = retriever.get_relevant_lessons(user_input_text, agent_name="critic")

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

    def get_validation_report(self):
        """Return latest validation report for UI display."""
        return self.validator.get_latest_report()

    def get_run_count(self):
        return self.run_count
