import sys
import traceback
sys.path.insert(0, ".")
from tests.test_phase1 import test_cloud_responds, test_cloud_structured_output
from model.cloud_manager import CloudManager

print("Testing CloudManager initialization...")
cm = CloudManager()

try:
    print("Running test_cloud_responds...")
    test_cloud_responds(cm)
    print("test_cloud_responds PASSED")
except Exception as e:
    print(f"test_cloud_responds FAILED: {e}")
    traceback.print_exc()

try:
    print("\\nRunning test_cloud_structured_output...")
    test_cloud_structured_output(cm)
    print("test_cloud_structured_output PASSED")
except Exception as e:
    print(f"test_cloud_structured_output FAILED: {e}")
    traceback.print_exc()
