"""Runs only after evaluate.py passed (DVC stops on gate failure). Copies the candidate into models/ (what the API/UI loads) and registers it in the MLflow Model Registry."""
import json, os, shutil, pathlib
c = pathlib.Path("models/candidate"); assert json.load(open("reports/gate.json"))["passed"], "gate not passed"
for f in ("classifier.pt", "classes.json", "calibration.json", "model_card.json"): shutil.copy(c / f, pathlib.Path("models") / f)
try:
    import mlflow; mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")); card = json.load(open("models/model_card.json"))
    v = mlflow.register_model(f"runs:/{card['mlflow_run_id']}/classifier.pt", "ecosort-classifier"); print("registered version", v.version)
except Exception as e: print("MLflow registry skipped:", e)
print("promoted", json.load(open("models/model_card.json"))["version"])
