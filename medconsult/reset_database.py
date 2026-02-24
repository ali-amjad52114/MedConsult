#!/usr/bin/env python3
"""
Robust database reset script for ChromaDB corruption issues.
Use this when you get 'no such table: tenants' errors.
"""

import os
import shutil
import sys

def reset_database():
    """Completely reset ChromaDB and cached modules."""
    
    print("🔧 Resetting MedConsult database...")
    
    # Step 1: Remove corrupted database
    db_path = 'experience_library/chroma_db'
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
            print(f"✅ Removed corrupted database: {db_path}")
        except Exception as e:
            print(f"❌ Failed to remove database: {e}")
            return False
    
    # Step 2: Remove cache directories
    cache_dirs = ['__pycache__', '.pytest_cache', '.ipynb_checkpoints']
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ Removed cache: {cache_dir}")
            except Exception as e:
                print(f"⚠️ Could not remove {cache_dir}: {e}")
    
    # Step 3: Clear Python module cache
    modules_to_clear = [
        'cloud_manager', 'pipeline', 'evaluator', 'lesson_extractor', 
        'sirius', 'agents', 'memory_store', 'memory_retriever',
        'augmentation', 'validator', 'medgemma_manager'
    ]
    
    cleared_count = 0
    for module_name in list(sys.modules.keys()):
        if any(target in module_name for target in modules_to_clear):
            del sys.modules[module_name]
            cleared_count += 1
    
    print(f"✅ Cleared {cleared_count} cached modules")
    
    # Step 4: Reset singletons
    try:
        from model.cloud_manager import CloudManager
        CloudManager._instance = None
        print("✅ Reset CloudManager singleton")
    except ImportError:
        print("⚠️ Could not reset CloudManager (not imported)")
    
    print()
    print("🎉 Database reset complete!")
    print("Now run your pipeline test again.")
    
    return True

if __name__ == "__main__":
    reset_database()
