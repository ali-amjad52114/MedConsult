# MedConsult

**Multi-agent medical analysis that gets smarter with every case.**

MedConsult is a transparent AI reasoning pipeline for medical data — lab reports, clinical notes, and medical images. Three specialized [MedGemma](https://huggingface.co/google/medgemma-1.5-4b-it) agents analyze data sequentially, while a background **SiriuS** self-improvement loop evaluates quality, extracts lessons, and stores them in a vector database so every future analysis is better informed.

> ⚕️ Built for the **MedGemma Impact Challenge 2026** — research demo, not a clinical device.

---

## Why MedConsult?

| Problem | Solution |
|---|---|
| AI "black box" reasoning | Three transparent agents with visible chain-of-thought |
| Static models that don't learn | SiriuS loop extracts lessons into persistent ChromaDB memory |
| Patient-unfriendly medical jargon | Critic agent rewrites findings in plain language |
| Cloud dependency for clinical data | MedGemma runs fully local; only meta-reasoning uses cloud |
| One-shot errors with no recovery | Augmentation re-runs low-scoring chains with escalating feedback |

---

## How It Works

```
  ┌──────────────────────────────────────────────┐
  │         USER UPLOADS MEDICAL DATA             │
  │     (lab report / clinical note / image)      │
  └─────────────────────┬────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │  Memory Retriever  │  ← injects relevant lessons
              │    (ChromaDB)      │     from past analyses
              └─────────┬──────────┘
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
 ┌───────────┐   ┌───────────┐   ┌───────────┐
 │  ANALYST  │ → │ CLINICIAN │ → │  CRITIC   │
 │ MedGemma  │   │ MedGemma  │   │ MedGemma  │
 │           │   │           │   │           │
 │ Extract   │   │ Interpret │   │ Review +  │
 │ findings  │   │ patterns  │   │ summarize │
 └───────────┘   └───────────┘   └─────┬─────┘
                                        │
                          ┌─────────────▼──── USER GETS RESULTS
                          │                   (5 output tabs)
                ┌─────────▼──────────┐
                │    EVALUATOR        │  ← async, non-blocking
                │  (Gemini Cloud)     │
                │  Scores chain 1–5   │
                └─────────┬──────────┘
                          │
              ┌───────────▼────────────┐
              │  Score ≤ 2?            │
              │  YES → Augmentation    │  re-run with feedback
              │  NO  → Continue        │
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │   LESSON EXTRACTOR     │  ← distills reasoning patterns
              │    (Gemini Cloud)      │
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │  ChromaDB Vector Store │  ← persists lessons for future
              └────────────────────────┘
```

---

## Key Features

### Three-Agent Reasoning Chain (Local, Private)
- **Analyst** — extracts factual findings, flags out-of-range values, supports lab reports and imaging
- **Clinician** — interprets clinical significance, identifies patterns and urgency levels
- **Critic** — reviews accuracy, rewrites findings in plain language for patients

### SiriuS Self-Improvement Loop (Background, Async)
Inspired by [Zhao et al. (2025)](https://arxiv.org/abs/2501.16243), the SiriuS loop runs after every analysis without blocking the user:

1. **Evaluate** — Gemini scores the full chain 1–5
2. **Augment** — Low scores (≤2) trigger a retry with escalating feedback strategies
3. **Extract** — Successful chains yield 2–5 structured reasoning lessons
4. **Store** — Lessons persist in ChromaDB with agent/type/confidence metadata
5. **Inject** — Relevant lessons appear in each agent's prompt at next inference

**No fine-tuning required.** The model weights never change — only the prompt context grows smarter.

### Gradio Web UI — 5 Output Tabs
| Tab | Content |
|---|---|
| Patient Summary | Plain-language Critic output |
| Analyst Extraction | Raw factual findings |
| Clinical Interpretation | Clinician output with urgency level |
| Processing Info | Timing, model info, input type, lessons available |
| SiriuS Intelligence | Score, lessons injected/extracted, augmentation status, validation trend |

### Hybrid Inference
- **Local (MedGemma 1.5 4B)** — all three clinical agents, runs on GPU or CPU
- **Cloud (Gemini)** — Evaluator, Lesson Extractor, Periodic Validator only

---

## Architecture

```
medconsult/
├── app.py                    # Gradio interface (entry point)
├── pipeline.py               # MedConsultPipeline orchestrator
│
├── agents/
│   ├── analyst.py            # Fact extraction
│   ├── clinician.py          # Pattern interpretation
│   ├── critic.py             # Review + patient summary
│   └── evaluator.py          # Quality scoring (Gemini)
│
├── model/
│   ├── medgemma_manager.py   # MedGemma singleton (GPU/CPU, 4-bit quant)
│   └── cloud_manager.py      # Gemini/OpenAI with fallback chain
│
├── sirius/
│   ├── memory_store.py       # ChromaDB vector store
│   ├── memory_retriever.py   # Formats lessons as prompt context
│   ├── lesson_extractor.py   # Gemini-powered lesson distillation
│   ├── augmentation.py       # Escalating retry strategies
│   ├── experience_library.py # Saves chains to good/ bad/ JSON
│   └── validator.py          # Periodic quality auditor
│
├── prompts/                  # System instructions per agent
├── tests/                    # Phase 0–6 test suite
├── experience_library/       # Runtime: chains + ChromaDB + stats
├── verify_setup.py
└── prepopulate_memory.py
```

Full walkthrough: [`medconsult/ARCHITECTURE.md`](medconsult/ARCHITECTURE.md)

---

## Quickstart

### 1. Requirements

- Python 3.10–3.13
- GPU recommended (VRAM ≥ 8 GB for 4-bit MedGemma); CPU supported
- Hugging Face account with [MedGemma access](https://huggingface.co/google/medgemma-1.5-4b-it)
- Google API key (for Gemini — Evaluator + Lesson Extractor)

### 2. Install

```bash
git clone https://github.com/ali-amjad52114/MedConsult.git
cd MedConsult/medconsult
pip install -r requirements.txt
```

### 3. Configure

```bash
export HF_TOKEN="your_huggingface_token"
export GOOGLE_API_KEY="your_google_api_key"
# Optional fallback:
export OPENAI_API_KEY="your_openai_api_key"
```

### 4. Verify Setup

```bash
python verify_setup.py
```

### 5. Pre-populate Memory (recommended before first demo)

Runs the pipeline on 10 synthetic cases to seed ChromaDB with lessons:

```bash
python prepopulate_memory.py
```

### 6. Launch

```bash
python app.py
# → http://localhost:7860
```

---

## Performance

| Environment | Model Load | Per-Agent Generation | Full Chain |
|---|---|---|---|
| GPU (16 GB, 4-bit) | ~1 min | ~30 sec | ~2 min |
| CPU (float32) | ~28 min | ~3–5 min | ~10–15 min |

> SiriuS evaluation runs asynchronously — users receive results before it completes.

---

## Tech Stack

| Component | Technology |
|---|---|
| Clinical agents | [MedGemma 1.5 4B-IT](https://huggingface.co/google/medgemma-1.5-4b-it) |
| Meta-reasoning | [Gemini 2.0/2.5 Flash](https://ai.google.dev/) |
| Vector memory | [ChromaDB](https://www.trychroma.com/) + `all-MiniLM-L6-v2` |
| Embedding | `sentence-transformers` |
| Quantization | `bitsandbytes` (nf4 4-bit) |
| UI | [Gradio](https://gradio.app/) |
| Inference | `transformers`, `accelerate`, `torch` |

---

## Running Tests

```bash
cd medconsult

# Phase 1 — agent behavior
python -m pytest tests/test_phase1.py -v -s

# Phase 2 — pipeline structure
python -m pytest tests/test_phase2.py -v -s

# Full suite (slow on CPU — model loads each session)
python -m pytest tests/ -v
```

Tests cover: per-agent output, pipeline structure, SiriuS evaluation, augmentation loops, lesson extraction, memory retrieval, and UI mocks.

---

## Self-Improvement Details

### Lesson Schema
Each extracted lesson is stored with:
```python
{
  "target_agent": "analyst | clinician | critic",
  "lesson_type":  "extraction_pattern | reasoning_chain | communication_tip | pitfall_warning",
  "topic":        "brief label (3–5 words)",
  "rule":         "actionable instruction (1–2 sentences)",
  "confidence":   "high | medium | low"
}
```

### Augmentation Strategies
When a chain scores ≤ 2, three retry passes escalate in intensity:
1. **Precise** — issue + targeted actionable fix
2. **Examples** — correct patterns demonstrated inline
3. **Expert Mode** — senior-clinician persona with explicit reasoning chain

Anti-lessons (failure patterns) are extracted from bad chains and stored as pitfall warnings.

### Periodic Validation
Every 5 chains, a Gemini auditor reviews recent results on 4 dimensions: accuracy, reasoning, communication, safety. Trends and concerns appear in the SiriuS Intelligence tab.

---

## Citations

```bibtex
@misc{medgemma2025,
  title  = {MedGemma: A Family of Medical Foundation Models},
  author = {Google},
  year   = {2025},
  url    = {https://huggingface.co/google/medgemma-1.5-4b-it}
}

@misc{zhao2025sirius,
  title  = {SiriuS: Self-Improving Multi-Agent Systems via Bootstrapped Reasoning},
  author = {Zhao et al.},
  year   = {2025},
  url    = {https://arxiv.org/abs/2501.16243}
}
```

---

## Disclaimer

> **This is an AI research prototype, not a medical device or diagnostic tool.**
> Output must not be used for clinical decision-making without review by a licensed healthcare professional.
> MedConsult does not store or transmit patient data. All clinical inference runs locally.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with MedGemma · SiriuS · ChromaDB · Gemini · Gradio
</p>
