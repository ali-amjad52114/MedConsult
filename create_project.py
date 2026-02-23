import os

base_dir = r"c:\AI\Projects\MedConsult\medconsult"
os.makedirs(base_dir, exist_ok=True)

dirs = [
    "agents",
    "prompts",
    "tests/test_data",
]
for d in dirs:
    os.makedirs(os.path.join(base_dir, d), exist_ok=True)

files = {
    "app.py": '\"\"\"\nMain Gradio application for MedConsult.\n\"\"\"\n\ndef launch_ui():\n    \"\"\"\n    Initializes and launches the Gradio web application for MedConsult.\n    \"\"\"\n    pass\n\nif __name__ == "__main__":\n    launch_ui()\n',
    "agents/__init__.py": "",
    "agents/analyst.py": '\"\"\"\nThe Analyst Agent in the MedConsult pipeline.\nResponsible for data extraction and normalization.\n\"\"\"\n\ndef extract_data(input_data):\n    \"\"\"\n    Extracts structured facts from raw input data.\n    \"\"\"\n    pass\n\ndef normalize_profile(extracted_facts):\n    \"\"\"\n    Normalizes the extracted facts into a unified patient profile.\n    \"\"\"\n    pass\n',
    "agents/clinician.py": '\"\"\"\nThe Clinician Agent in the MedConsult pipeline.\nResponsible for generating differential diagnoses.\n\"\"\"\n\ndef generate_hypotheses(patient_profile):\n    \"\"\"\n    Generates hypotheses based on the normalized patient profile.\n    \"\"\"\n    pass\n\ndef outline_plan(hypotheses):\n    \"\"\"\n    Outlines a preliminary treatment or testing plan.\n    \"\"\"\n    pass\n',
    "agents/critic.py": '\"\"\"\nThe Critic Agent in the MedConsult pipeline.\nResponsible for reviewing the Clinician\'s work.\n\"\"\"\n\ndef review_diagnosis(clinician_output, raw_data):\n    \"\"\"\n    Reviews the clinician\'s output against the raw data for fallacies or missing cases.\n    \"\"\"\n    pass\n',
    "pipeline.py": '\"\"\"\nThe main workflow pipeline that chains the Analyst, Clinician, and Critic together.\n\"\"\"\n\ndef run_pipeline(input_data):\n    \"\"\"\n    Executes the full analyzing, diagnosing, and reviewing timeline.\n    \"\"\"\n    pass\n',
    "prompts/__init__.py": "",
    "prompts/analyst_prompt.py": '\"\"\"\nSystem prompts for the Analyst Agent.\n\"\"\"\n\nANALYST_SYSTEM_PROMPT = ""\n',
    "prompts/clinician_prompt.py": '\"\"\"\nSystem prompts for the Clinician Agent.\n\"\"\"\n\nCLINICIAN_SYSTEM_PROMPT = ""\n',
    "prompts/critic_prompt.py": '\"\"\"\nSystem prompts for the Critic Agent.\n\"\"\"\n\nCRITIC_SYSTEM_PROMPT = ""\n',
    "tests/test_phase0.py": 'import os\nimport pytest\n\ndef test_imports():\n    \"\"\"Verify all required packages can be imported.\"\"\"\n    try:\n        import torch\n        import transformers\n        import gradio\n        import PIL\n    except ImportError as e:\n        pytest.fail(f"Failed to import required package: {e}")\n\ndef test_project_structure():\n    \"\"\"Verify all project files and folders exist.\"\"\"\n    required_paths = [\n        "app.py",\n        "agents/__init__.py",\n        "agents/analyst.py",\n        "agents/clinician.py",\n        "agents/critic.py",\n        "pipeline.py",\n        "prompts/__init__.py",\n        "prompts/analyst_prompt.py",\n        "prompts/clinician_prompt.py",\n        "prompts/critic_prompt.py",\n        "tests/test_phase0.py",\n        "tests/test_phase1.py",\n        "tests/test_phase2.py",\n        "tests/test_phase3.py",\n        "tests/test_phase4.py",\n        "tests/test_phase5.py",\n        "tests/test_data/sample_cbc.txt",\n        "tests/test_data/sample_clinical_note.txt",\n        "tests/test_data/README.md",\n        "requirements.txt",\n        "README.md"\n    ]\n    for path in required_paths:\n        assert os.path.exists(path), f"Path does not exist: {path}"\n\ndef test_sample_data_exists():\n    \"\"\"Verify both sample test data files exist and are non-empty.\"\"\"\n    cbc_path = "tests/test_data/sample_cbc.txt"\n    clin_path = "tests/test_data/sample_clinical_note.txt"\n    assert os.path.exists(cbc_path) and os.path.getsize(cbc_path) > 100, f"File missing or too small: {cbc_path}"\n    assert os.path.exists(clin_path) and os.path.getsize(clin_path) > 100, f"File missing or too small: {clin_path}"\n\ndef test_gpu_available():\n    \"\"\"Verify CUDA GPU is accessible, warn if not.\"\"\"\n    import torch\n    import warnings\n    if torch.cuda.is_available():\n        print(f"\\nGPU available: {torch.cuda.get_device_name(0)}")\n    else:\n        warnings.warn("No GPU available, falling back to CPU. This may be slow.")\n',
    "tests/test_phase1.py": "",
    "tests/test_phase2.py": "",
    "tests/test_phase3.py": "",
    "tests/test_phase4.py": "",
    "tests/test_phase5.py": "",
    "tests/test_data/README.md": "# Test Data\nContains sample inputs for evaluating agents.",
    "README.md": "# MedConsult\nA transparent medical reasoning multi-agent system.",
    "requirements.txt": "torch\ntransformers\naccelerate\ngradio\nPillow\npytest\n",
    "tests/test_data/sample_cbc.txt": \"\"\"---\nCOMPLETE BLOOD COUNT (CBC) REPORT\nPatient: John Doe | Age: 52 | Sex: Male | Date: 2026-02-15\n\nTest                  Result    Unit        Reference Range    Flag\nWhite Blood Cells     12.8      x10^9/L     4.0 - 11.0        HIGH\nRed Blood Cells       4.2       x10^12/L    4.5 - 5.5         LOW\nHemoglobin            11.2      g/dL        13.5 - 17.5       LOW\nHematocrit            33.8      %           38.5 - 50.0       LOW\nMCV                   80.5      fL          80.0 - 100.0      NORMAL\nMCH                   26.7      pg          27.0 - 33.0       LOW\nMCHC                  33.1      g/dL        31.5 - 35.5       NORMAL\nPlatelets             415       x10^9/L     150 - 400         HIGH\nNeutrophils           78.2      %           40.0 - 70.0       HIGH\nLymphocytes           14.1      %           20.0 - 40.0       LOW\n---\n\"\"\",
    "tests/test_data/sample_clinical_note.txt": \"\"\"---\nDISCHARGE SUMMARY\nPatient: Jane Smith | Age: 67 | Sex: Female\nAdmission Date: 2026-02-10 | Discharge Date: 2026-02-14\n\nCHIEF COMPLAINT: Shortness of breath and bilateral leg swelling for 3 days.\n\nHISTORY OF PRESENT ILLNESS: 67-year-old female with known history of hypertension and type 2 diabetes mellitus presented with progressive dyspnea on exertion and bilateral lower extremity edema over the past 3 days. She reports sleeping with 3 pillows and occasional paroxysmal nocturnal dyspnea. She denies chest pain, palpitations, or syncope. She admits to dietary indiscretion over the holidays and missed her furosemide for 4 days.\n\nVITAL SIGNS ON ADMISSION: BP 162/94, HR 96, RR 22, SpO2 92% on room air, Temp 37.1C\n\nPHYSICAL EXAM: JVP elevated to 10cm. Bilateral crackles at lung bases. S3 gallop present. 2+ bilateral pitting edema to mid-shin.\n\nLABS: BNP 1840 pg/mL (normal <100). Creatinine 1.8 mg/dL (baseline 1.2). Troponin negative x2. Chest X-ray showing bilateral pleural effusions and pulmonary congestion.\n\nHOSPITAL COURSE: Patient was diagnosed with acute decompensated heart failure, likely precipitated by medication non-compliance and dietary indiscretion. She was treated with IV furosemide with good diuretic response. She lost 4.2 kg during admission. Renal function improved with creatinine trending down to 1.4 by discharge. She was restarted on her home medications including lisinopril 20mg, metformin 1000mg BID, and furosemide 40mg daily (increased from 20mg). Cardiology was consulted and recommended outpatient echocardiogram.\n\nDISCHARGE DIAGNOSIS: Acute decompensated heart failure (HFpEF). Hypertension. Type 2 diabetes mellitus.\n\nDISCHARGE INSTRUCTIONS: Low sodium diet (<2g daily). Daily weights. Return if weight gain >2kg in 2 days. Follow up with cardiology in 1 week. Follow up with PCP in 2 weeks.\n---\n\"\"\"
}

for fname, content in files.items():
    path = os.path.join(base_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("Project files created successfully.")
