import sys
import traceback

def run():
    try:
        from tests import test_phase1
        print("Running test_model_loads...")
        test_phase1.test_model_loads()
        
        print("Running test_text_generation...")
        test_phase1.test_text_generation()
        
        print("Running test_follows_system_prompt...")
        test_phase1.test_follows_system_prompt()
        
        print("Running test_different_prompts_different_outputs...")
        test_phase1.test_different_prompts_different_outputs()
        
        print("ALL TESTS PASSED MANUALLY")
    except Exception as e:
        print("ERROR RUNNING TESTS:")
        traceback.print_exc()

if __name__ == "__main__":
    run()
