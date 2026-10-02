"""Held-out TEST evaluation + promotion gate. Exits non-zero (blocking promotion/deploy) if the candidate fails any check."""
import sys, json, os, pathlib, yaml, numpy as np, torch
sys.path.insert(0, "src")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from ecosort.preprocess import eval_tf
from ecosort.evalutils import softmax, ece, macro_f1, weighted_f1, accuracy, confusion, f1_per_class, pair_rates
P = yaml.safe_load(open("params.yaml")); E = P["evaluate"]; cand = pathlib.Path("models/candidate"); card = json.load(open(cand / "model_card.json")); classes = json.load(open(cand / "classes.json"))
T = json.load(open(cand / "calibration.json"))["temperature"]; ds = datasets.ImageFolder("data/processed/test", eval_tf(card["img_size"])); assert ds.classes == classes, "test classes differ from model classes"
m = torch.jit.load(str(cand / "classifier.pt")).eval(); zs, ys = [], []
with torch.no_grad():
    for x, y in DataLoader(ds, 64): zs.append(m(x).numpy()); ys.append(y.numpy())
z, y = np.concatenate(zs), np.concatenate(ys); pr = softmax(z, T); pred = pr.argmax(1); n = len(classes); cm = confusion(y, pred, n); pairs = pair_rates(cm, classes, P["watch_pairs"], E["min_pair_support"])
res = dict(version=card["version"], arch=card["arch"], n_test=int(len(y)), accuracy=accuracy(y, pred), macro_f1=macro_f1(y, pred, n), weighted_f1=weighted_f1(y, pred, n), ece_calibrated=ece(pr, y), ece_uncalibrated=ece(softmax(z), y),
           per_class_f1={c: (None if np.isnan(f) else float(f)) for c, f in zip(classes, f1_per_class(cm))}, confusable_pairs=pairs)
checks = {"macro_f1 >= min": res["macro_f1"] >= E["min_macro_f1"], "ece <= max": res["ece_calibrated"] <= E["max_ece"],
          "confusable pairs <= max rate": all(p["rate"] <= E["max_confusable_rate"] for p in pairs if p["evaluated"])}
prod = pathlib.Path("models/model_card.json")
if prod.exists() and (pf := json.load(open(prod)).get("test_macro_f1")) is not None: checks["not worse than production"] = res["macro_f1"] >= pf - E["regression_tolerance"]
res["gate"] = {k: bool(v) for k, v in checks.items()}; res["passed"] = all(checks.values())
fig, ax = plt.subplots(figsize=(1 + .7 * n, 1 + .6 * n)); ax.imshow(cm, cmap="Greens"); ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(classes, rotation=75, ha="right"); ax.set_yticklabels(classes)
for i in range(n):
    for j in range(n): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.tight_layout(); fig.savefig("reports/confusion_matrix.png", dpi=130)
json.dump(res, open("metrics.json", "w"), indent=2); json.dump(res, open("reports/gate.json", "w"), indent=2)
card.update(test_macro_f1=res["macro_f1"], test_weighted_f1=res["weighted_f1"], test_accuracy=res["accuracy"], test_ece=res["ece_calibrated"]); json.dump(card, open(cand / "model_card.json", "w"), indent=2)
try:
    import mlflow; mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    with mlflow.start_run(run_id=card["mlflow_run_id"]): mlflow.log_metrics({"test_macro_f1": res["macro_f1"], "test_accuracy": res["accuracy"], "test_ece": res["ece_calibrated"]}); mlflow.log_artifact("reports/confusion_matrix.png"); mlflow.set_tag("gate_passed", res["passed"])
except Exception as e: print("mlflow logging skipped:", e)
print(json.dumps({k: v for k, v in res.items() if k != "per_class_f1"}, indent=2)); sys.exit(0 if res["passed"] else 1)
