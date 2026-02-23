# MedConsult — Multi-Agent Medical Analysis

A transparent medical reasoning multi-agent system with **SiriuS** self-improvement.  
Three MedGemma agents analyze medical data; Gemini evaluates quality and extracts lessons for persistent in-context learning.

**MedGemma Impact Challenge 2026**

📖 **ARCHITECTURE.md** — File-by-file guide, orchestration flowchart, data flow.

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    USER UPLOADS MEDICAL DATA                 │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Memory Retriever │ ← retrieves relevant lessons
                    │  (ChromaDB)       │   from past analyses
                    └────────┬─────────┘
                             │ injects lessons into prompts
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │   ANALYST     │→│  CLINICIAN    │→│   CRITIC      │
  │  (MedGemma)   │  │  (MedGemma)   │  │  (MedGemma)   │
  │  Extract facts │  │  Interpret    │  │  Review +      │
  │               │  │  patterns     │  │  Patient summary│
  └──────────────┘  └──────────────┘  └──────┬───────┘
                                              │
                              ┌────────────────▼──── USER GETS RESULTS
                              │
                    ┌─────────▼─────────┐
                    │    EVALUATOR       │  ← runs ASYNC in background
                    │   (Gemini Cloud)   │
                    │   Scores chain 1-5 │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Score ≤ 2?         │
                    │ YES → Augmentation │  re-run with feedback
                    │ NO  → Continue     │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ LESSON EXTRACTOR   │  ← distills reasoning patterns
                    │ (Gemini Cloud)     │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   ChromaDB         │  ← stores lessons for future
                    │   Vector Store     │     analyses (persistent memory)
                    └───────────────────┘
```

---

## Setup

### Requirements

- **Python 3.10–3.13** (3.14 may have ChromaDB compatibility issues)
- **GPU** recommended for MedGemma (CPU fallback supported, slower)
- **Hugging Face token** for MedGemma
- **Google API key** for Gemini (Evaluator, Lesson Extractor)

### Install

```bash
cd medconsult
pip install -r requirements.txt
```

### Environment

```bash
# Hugging Face (for MedGemma)
export HF_TOKEN="your_hf_token"

# Google AI (for Evaluator + Lesson Extractor)
export GOOGLE_API_KEY="your_google_api_key"
```

### Verify

```bash
python verify_setup.py
```

### Pre-populate memory (before demo)

Run the pipeline on all test inputs to build 10–15 lessons in ChromaDB:

```bash
python prepopulate_memory.py
```

### Launch

```bash
python app.py
```

Opens Gradio at `http://0.0.0.0:7860` with `share=True`.

---

## SiriuS Framework

MedConsult implements **SiriuS** (Zhao et al., 2025) for self-improvement:

1. **Evaluate** — Gemini scores each analysis chain (1–5).
2. **Augment** — If score ≤ 2, agents re-run with evaluator feedback.
3. **Extract** — Successful chains yield reasoning lessons.
4. **Store** — Lessons go into ChromaDB for retrieval.
5. **Inject** — At inference, relevant lessons are injected into each agent’s context.

We use **persistent in-context learning** instead of supervised fine-tuning to keep MedGemma weights stable while improving behavior through prompt augmentation.

---

## Citations

- **MedGemma:** [Google MedGemma](https://huggingface.co/google/medgemma-1.5-4b-it)
- **SiriuS:** Zhao et al. (2025). *SiriuS: Self-Improvement with Retrieval and Self-Supervised Learning.*
- **ChromaDB:** [ChromaDB](https://www.trychroma.com/)
- **Gemini:** [Google AI](https://ai.google.dev/)

---

## Documentation

- **ARCHITECTURE.md** — File-by-file explanation, orchestration flowchart, and data flow.

## Project Structure

```
medconsult/
├── app.py              # Gradio web interface
├── pipeline.py         # MedConsultPipeline (Analyst → Clinician → Critic)
├── agents/
│   ├── analyst.py      # Fact extraction (MedGemma)
│   ├── clinician.py    # Pattern interpretation (MedGemma)
│   ├── critic.py       # Patient summary + review (MedGemma)
│   └── evaluator.py    # Quality scoring (Gemini)
├── model/
│   ├── medgemma_manager.py
│   └── cloud_manager.py
├── sirius/
│   ├── memory_store.py    # ChromaDB vector store
│   ├── memory_retriever.py
│   ├── lesson_extractor.py
│   ├── augmentation.py    # Retry with feedback
│   └── experience_library.py
├── tests/
├── results/            # Saved example outputs
├── experience_library/ # Raw chains + ChromaDB
├── verify_setup.py
└── prepopulate_memory.py
```

---

## License & Disclaimer

⚠️ **AI research demo, not a medical device.**  
Always consult a healthcare provider. Do not use for clinical decisions.
