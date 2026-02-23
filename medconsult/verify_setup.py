#!/usr/bin/env python3
"""
Verify MedConsult setup: Python, deps, GPU, API keys, and quick inference.
Run before demo to ensure environment is ready.
"""

import sys
import os


def check_python():
    """Python 3.10+ required."""
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    print(f"Python {v.major}.{v.minor}.{v.micro}: {'OK' if ok else 'FAIL (need 3.10+)'}")
    return ok


def check_deps():
    """Core dependencies present."""
    deps = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("gradio", "gradio"),
        ("PIL", "PIL"),
        ("chromadb", "chromadb"),
        ("google.generativeai", "google.generativeai"),
    ]
    ok = True
    for name, mod in deps:
        try:
            __import__(mod)
            print(f"  {name}: OK")
        except ImportError as e:
            print(f"  {name}: FAIL - {e}")
            ok = False
    return ok


def check_gpu():
    """CUDA available for MedGemma."""
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            print(f"GPU: OK ({torch.cuda.get_device_name(0)})")
        else:
            print("GPU: WARN (CPU only, slower)")
        return True  # CPU fallback OK
    except Exception as e:
        print(f"GPU: FAIL - {e}")
        return False


def check_api_keys():
    """HF_TOKEN and GOOGLE_API_KEY."""
    hf = os.environ.get("HF_TOKEN") or os.path.exists(
        os.path.expanduser("~/.huggingface/token")
    )
    google = os.environ.get("GOOGLE_API_KEY")
    print(f"HF_TOKEN: {'OK' if hf else 'MISSING'}")
    print(f"GOOGLE_API_KEY: {'OK' if google else 'MISSING'}")
    return bool(hf) and bool(google)


def check_quick_inference():
    """Quick MedGemma inference (skip if slow)."""
    print("Quick inference: running...")
    try:
        from model.medgemma_manager import MedGemmaManager
        from agents.analyst import AnalystAgent
        from prompts.analyst_prompt import ANALYST_SYSTEM_PROMPT

        mgr = MedGemmaManager()
        analyst = AnalystAgent(mgr)
        out = analyst.analyze("Glucose: 100 mg/dL (reference 70-100)")
        ok = len(out) > 10
        print(f"Quick inference: {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        print(f"Quick inference: FAIL - {e}")
        return False


def main():
    print("=== MedConsult Setup Verification ===\n")
    results = []
    results.append(("Python", check_python()))
    print()
    print("Dependencies:")
    results.append(("Deps", check_deps()))
    print()
    results.append(("GPU", check_gpu()))
    print()
    results.append(("API keys", check_api_keys()))
    print()
    # Option: skip inference if --fast
    if "--fast" not in sys.argv:
        results.append(("Inference", check_quick_inference()))
    else:
        print("Quick inference: SKIPPED (--fast)")
        results.append(("Inference", True))

    print("\n=== Summary ===")
    all_ok = all(r[1] for r in results)
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
