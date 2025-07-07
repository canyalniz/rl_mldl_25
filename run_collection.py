import glob
from pathlib import Path
import subprocess

model_files = glob.glob("logs_and_models/source_optuna_2/*.zip")
for mf in model_files:
    model = Path(mf).stem
    if not ("target" in model):
        subprocess.run(["python", "collect_transitions.py", "--episodes", "2000", "--model-name", model])