# MedConsult Technical Specification

## 1. Product Summary

MedConsult is a multi-agent medical analysis application with a two-tier inference architecture:

- Clinical tier (local): MedGemma 1.5 4B (`google/medgemma-1.5-4b-it`) for patient-facing analysis.
- Meta tier (cloud): Gemini/OpenAI-compatible cloud model for quality evaluation and lesson extraction.

The core runtime chain is:

1. Analyst (fact extraction)
2. Clinician (clinical interpretation)
3. Critic (review + patient-friendly communication)

After user results are returned, a background self-improvement loop evaluates quality, stores trajectories, and updates persistent memory (ChromaDB) for future in-context retrieval.

## 2. Primary Objectives

- Provide structured medical analysis from clinical text and optional medical images.
- Keep patient-facing reasoning on a medical model (MedGemma).
- Improve over time through trajectory scoring, augmentation retries, and lesson memory.
- Maintain a responsive UX by running self-improvement asynchronously.

## 3. Implemented Architecture

### 3.1 High-level Components

- UI and interaction
  - `medconsult/app.py`
  - `medconsult/optimized_app.py`
- Pipeline orchestration
  - `medconsult/pipeline.py`
  - `medconsult/optimized_pipeline.py`
- Model managers
  - `medconsult/model/medgemma_manager.py`
  - `medconsult/model/cloud_manager.py`
  - `medconsult/model/model_manager.py` (legacy-compatible manager still used in some tests)
- Clinical agents
  - `medconsult/agents/analyst.py`
  - `medconsult/agents/clinician.py`
  - `medconsult/agents/critic.py`
- Meta agent
  - `medconsult/agents/evaluator.py`
- Self-improvement and memory
  - `medconsult/sirius/experience_library.py`
  - `medconsult/sirius/memory_store.py`
  - `medconsult/sirius/memory_retriever.py`
  - `medconsult/sirius/lesson_extractor.py`
  - `medconsult/sirius/augmentation.py`

### 3.2 Runtime Sequence (Current Code)

1. Input arrives via Gradio (`_process` in `app.py`).
2. Pipeline classifies input type (`ExperienceLibrary.classify_input_type`).
3. Pipeline retrieves relevant memory context for each agent from ChromaDB.
4. Analyst runs on MedGemma.
5. Clinician runs on MedGemma with analyst output.
6. Critic runs on MedGemma with full prior chain.
7. User receives all three outputs plus metadata/timing.
8. In background, evaluator scores chain quality.
9. If score <= 2, augmentation retries with targeted feedback.
10. Chain is stored in JSON experience library (`good/` vs `bad/`).
11. If score >= 3, lesson extractor distills reasoning lessons and stores them in ChromaDB.
12. Next runs use retrieved lessons as prompt context.

## 4. Module Behavior Details

### 4.1 MedGemmaManager

File: `medconsult/model/medgemma_manager.py`

- Singleton model loader.
- Loads MedGemma processor + model from Hugging Face.
- Uses 4-bit quantization on GPU via bitsandbytes:
  - `load_in_4bit=True`
  - `bnb_4bit_quant_type="nf4"`
- CPU fallback supported.
- Supports text-only and image+text inference.
- Chat template is applied through `AutoProcessor.apply_chat_template`.

### 4.2 CloudManager

File: `medconsult/model/cloud_manager.py`

- Singleton cloud inference manager.
- Provider selection:
  - Primary: Google Generative AI via `GOOGLE_API_KEY`.
  - Fallback: OpenAI via `OPENAI_API_KEY`.
- Gemini model fallback list implemented (`gemini-2.5-flash`, `gemini-2.0-flash-001`, etc.).
- Retry with backoff and token usage counters.
- Used only for meta tasks (evaluator + lesson extraction), not patient-facing clinical reasoning.

### 4.3 AnalystAgent

File: `medconsult/agents/analyst.py`

- Injects optional memory context and evaluator feedback.
- Extracts factual medical findings (prompt-constrained).
- Uses up to 2048 tokens for text, 512 when image is present.

### 4.4 ClinicianAgent

File: `medconsult/agents/clinician.py`

- Consumes original input + analyst output.
- Injects optional memory context and feedback.
- Produces pattern-level interpretation, differential considerations, urgency.

### 4.5 CriticAgent

File: `medconsult/agents/critic.py`

- Consumes original input + analyst + clinician outputs.
- Produces:
  - critical review of reasoning quality
  - patient-friendly summary with explicit disclaimer.

### 4.6 EvaluatorAgent

File: `medconsult/agents/evaluator.py`

- Sends full chain to cloud model using strict evaluation prompt.
- Parses:
  - score (1-5)
  - key issues list
  - suggested improvements list
- Defaults to score=3 if parsing is incomplete.

### 4.7 ExperienceLibrary

File: `medconsult/sirius/experience_library.py`

- Persists full chain JSON to:
  - `experience_library/good/` for score >= 3
  - `experience_library/bad/` for score < 3
- Tracks stats in `experience_library/stats.json`.
- Input classification is keyword-based.

### 4.8 MemoryStore

File: `medconsult/sirius/memory_store.py`

- ChromaDB persistent collection `medical_lessons`.
- Stores lesson rules as embedded documents, metadata includes:
  - topic
  - input_type
  - confidence
  - source_score
  - chain_id
- Retrieval supports optional input type filtering.

### 4.9 MemoryRetriever

File: `medconsult/sirius/memory_retriever.py`

- Fetches top lessons and formats them as structured context block.
- If context gets long, truncates to top 3 lessons.
- Returns `None` when no lessons are available.

### 4.10 LessonExtractor

File: `medconsult/sirius/lesson_extractor.py`

- Uses cloud model to convert successful chains into reusable reasoning lessons.
- Parses XML-like `<lesson>` blocks.
- Returns normalized lesson dictionaries for storage.

### 4.11 AugmentationLoop

File: `medconsult/sirius/augmentation.py`

- Triggered when evaluator score <= 2.
- Builds feedback from issues/improvements.
- Re-runs Analyst->Clinician->Critic with feedback injected.
- Re-evaluates and accepts improved result when:
  - score >= 4, or
  - score improves from prior attempt.

## 5. UI Specification

File: `medconsult/app.py`

- Gradio app with:
  - Text input
  - Optional image upload
  - Submit/Clear controls
- Output tabs:
  - Patient Summary (Critic)
  - Analyst Extraction
  - Clinical Interpretation
  - Processing Info (JSON metadata + timing)
  - SiriuS Intelligence (score, augmentation, memory stats)
- UX design choice:
  - User response path is non-blocking for self-improvement.
  - Background learning is bounded with thread join timeout.

## 6. Data Contracts

### 6.1 Pipeline Result

Approximate shape returned by `run()` and `run_with_timing()`:

```json
{
  "input": "string",
  "analyst": "string",
  "clinician": "string",
  "critic": "string",
  "timings": {
    "analyst": 0.0,
    "clinician": 0.0,
    "critic": 0.0,
    "total": 0.0
  },
  "metadata": {
    "model": "google/medgemma-1.5-4b-it",
    "meta_model": "gemini-2.5-flash",
    "pipeline_version": "4.0-sirius-cloud",
    "timestamp": "ISO8601",
    "input_type": "lab_report|clinical_note|imaging|general",
    "agent_chain": ["analyst", "clinician", "critic"],
    "memory_used": {
      "analyst": true,
      "clinician": false,
      "critic": true
    },
    "total_lessons_available": 12
  }
}
```

### 6.2 Evaluator Result

```json
{
  "score": 4,
  "raw_evaluation": "string",
  "issues": ["..."],
  "improvements": ["..."]
}
```

### 6.3 Lesson Object

```json
{
  "topic": "string",
  "input_type": "lab_report|clinical_note|imaging|general",
  "rule": "string",
  "example_values": "string",
  "confidence": "high|medium",
  "source_score": 4,
  "chain_id": "string"
}
```

## 7. Testing Strategy

Test files: `medconsult/tests/test_phase0.py` ... `medconsult/tests/test_phase6.py`

Coverage areas include:

- setup and dependency checks
- model manager behavior
- per-agent behavioral checks
- full pipeline output structure and timing
- self-improvement and memory storage/retrieval
- UI interaction flow using mocks

## 8. Performance/Scalability Notes

- Current pipeline execution is sequential for clinical agents due to dependency chain.
- `optimized_pipeline.py` parallelizes memory retrieval for minor speed gains.
- Most latency is generated by MedGemma token generation length and model load/state.
- Self-improvement is run post-response to protect UX responsiveness.
- Persistent memory grows over time; retrieval quality depends on lesson quality and embeddings.

## 9. Security and Operations Requirements

- Required environment variables:
  - `HF_TOKEN` for MedGemma downloads/access
  - `GOOGLE_API_KEY` (or `OPENAI_API_KEY`) for cloud meta-model calls
- No API keys should be hard-coded in source files.
- ChromaDB and experience JSON are local persistent artifacts and should be handled as sensitive when containing clinical content.

## 10. Known Technical Gaps

1. Stale module risk:
- `medconsult/sirius/orchestrator.py` references functions not present in current Sirius modules and is not the active pipeline path.

2. Mixed model-manager usage:
- `medconsult/model/model_manager.py` (legacy manager) is still used in some tests while production path uses `medgemma_manager.py`.

3. Test drift:
- `medconsult/tests/test_phase4b.py` contains calls (e.g., `critic.summarize`) that do not match current `CriticAgent` interface (`review_and_communicate`), indicating partial legacy drift.

4. Strict SiriuS parity:
- Current system is SiriuS-inspired and retrieval-centric.
- It does not yet implement iterative agent weight updates (SFT/LoRA) from trajectory datasets.

## 11. Roadmap

### Near-term

1. Unify on one local model manager and remove legacy code paths.
2. Align all tests with current agent interfaces.
3. Add explicit schema validation for evaluator and lesson parser outputs.
4. Improve timeout and fallback behavior for cloud meta calls.

### Mid-term

1. Add robust observability:
  - structured logs
  - per-stage latency histograms
  - cloud cost telemetry.
2. Improve retrieval quality:
  - domain-adapted embeddings
  - relevance filtering by agent role and confidence.

### Long-term

1. Add strict SiriuS training loop:
  - build trajectory dataset
  - per-agent LoRA fine-tuning
  - checkpointed iterative improvement cycle.
2. Add deployment profiles:
  - local demo mode
  - cloud-scaled inference mode with queue workers.

## 12. Compliance/Clinical Disclaimer

MedConsult is a research demonstration system, not a regulated medical device. Outputs are assistive and must not replace clinical judgment or licensed medical advice.
