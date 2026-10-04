"""Traceability ledger (hash-chained), feedback store, analytics & forecasting."""
import json, os, time, hashlib, numpy as np, pandas as pd
LEDGER = os.getenv("ECOSORT_LEDGER", "data/ledger.jsonl"); FEEDBACK = "data/feedback/feedback.jsonl"

def ledger_append(rec, path=LEDGER):
    prev = "0" * 64
    if os.path.exists(path) and (ls := [l for l in open(path).read().splitlines() if l.strip()]): prev = json.loads(ls[-1])["hash"]
    rec = {**rec, "prev": prev, "ts": round(time.time(), 2)}
    rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    dir_name = os.path.dirname(path)
    if dir_name: os.makedirs(dir_name, exist_ok=True)
    open(path, "a").write(json.dumps(rec) + "\n"); return rec

def ledger_verify(path=LEDGER):
    prev = "0" * 64
    for l in [l for l in open(path).read().splitlines() if l.strip()] if os.path.exists(path) else []:
        r = json.loads(l); h = r.pop("hash")
        if r["prev"] != prev or hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() != h: return False
        prev = h
    return True

def save_feedback(image_name, box, predicted, corrected):
    os.makedirs(os.path.dirname(FEEDBACK), exist_ok=True)
    open(FEEDBACK, "a").write(json.dumps(dict(image=image_name, box=box, predicted=predicted, corrected=corrected, ts=time.time())) + "\n")

def review_needed(conf, entropy, hazard, thr=0.6):
    return bool(conf < thr or entropy > 0.8 or hazard)

def analytics(items):
    if not items:
        return dict(items=0, kg=0.0, recovered_kg=0.0, diversion_pct=0.0, review_pct=0.0)
    df = pd.DataFrame(items)
    df["recovered_kg"] = df.kg * df.recovery
    return dict(items=len(df), kg=round(df.kg.sum(), 2), recovered_kg=round(df.recovered_kg.sum(), 2),
                diversion_pct=round(100 * df.recovered_kg.sum() / max(df.kg.sum(), 1e-9), 1), review_pct=round(100 * df.review.mean(), 1))

def forecast(days=7, alpha=0.4, beta=0.2, seed=0):
    """Holt linear smoothing on daily waste-kg history (synthetic demo history; plug real ledger aggregates)."""
    rng = np.random.default_rng(seed); t = np.arange(60)
    y = 400 + 2 * t + 40 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 15, 60)
    l, b = y[0], y[1] - y[0]
    for v in y[1:]: l, b = alpha * v + (1 - alpha) * (l + b), beta * (alpha * v + (1 - alpha) * (l + b) - l) + (1 - beta) * b
    return pd.Series(y, name="history"), pd.Series([l + (i + 1) * b for i in range(days)], name="forecast")
