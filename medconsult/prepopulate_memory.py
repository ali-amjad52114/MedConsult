#!/usr/bin/env python3
"""
Pre-populate ChromaDB memory with lessons from all test inputs.
Run BEFORE demo so memory is active (10–15 lessons) and metadata shows memory_used=True.
"""

import os
import sys


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "tests", "test_data")
    inputs = []

    # CBC
    cbc_path = os.path.join(data_dir, "sample_cbc.txt")
    if os.path.exists(cbc_path):
        with open(cbc_path, "r") as f:
            inputs.append(("CBC", f.read()))

    # Clinical note
    clinical_path = os.path.join(data_dir, "sample_clinical_note.txt")
    if os.path.exists(clinical_path):
        with open(clinical_path, "r") as f:
            inputs.append(("Clinical", f.read()))

    # Short glucose
    inputs.append(("Glucose", "Glucose: 250 mg/dL (reference: 70-100)"))

    if not inputs:
        print("No test inputs found.")
        sys.exit(1)

    print("=== Pre-populating MedConsult Memory ===\n")

    try:
        from pipeline import MedConsultPipeline
    except Exception as e:
        print(f"Pipeline import failed: {e}")
        sys.exit(1)

    pipeline = MedConsultPipeline()
    total_lessons_before = pipeline.memory_store.get_lesson_count()
    print(f"Lessons before: {total_lessons_before}\n")

    for name, text in inputs:
        print(f"Processing: {name}...")
        try:
            result = pipeline.run_with_timing(text, image=None)
            sirius_result = pipeline.evaluate_and_learn(result, image=None)
            lessons = sirius_result.get("lessons_extracted", 0)
            total = sirius_result.get("total_lessons", 0)
            print(f"  -> lessons extracted: {lessons}, total in memory: {total}")
        except Exception as e:
            print(f"  -> FAIL: {e}")

    total_lessons_after = pipeline.memory_store.get_lesson_count()
    print(f"\nLessons after: {total_lessons_after}")
    print(f"Total lessons learned: {total_lessons_after - total_lessons_before}")
    print("\nPre-population complete. Memory is ready for demo.")


if __name__ == "__main__":
    main()
