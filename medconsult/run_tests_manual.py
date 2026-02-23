import sys
import traceback

def run():
    try:
        from tests import test_phase0
        print("Running test_imports...")
        test_phase0.test_imports()
        
        print("Running test_project_structure...")
        test_phase0.test_project_structure()
        
        print("Running test_sample_data_exists...")
        test_phase0.test_sample_data_exists()
        
        print("Running test_gpu_available...")
        test_phase0.test_gpu_available()
        
        print("ALL TESTS PASSED MANUALLY")
    except Exception as e:
        print("ERROR RUNNING TESTS:")
        traceback.print_exc()

if __name__ == "__main__":
    run()
