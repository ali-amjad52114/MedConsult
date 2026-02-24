import subprocess

result = subprocess.run(
    [R"c:\AI\Projects\MedConsult\.venv\Scripts\pytest.exe", "medconsult/tests/test_phase4b.py", "-v"],
    cwd=R"c:\AI\Projects\MedConsult",
    capture_output=True,
    text=True
)

print("--- STDOUT ---")
print(result.stdout)
print("--- STDERR ---")
print(result.stderr)
