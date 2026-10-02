"""data/raw/<dataset>/<class>/*.jpg (+ validated human-review crops) -> data/processed/{train,val,test}.
Skips corrupted files, removes exact + near-duplicates (md5 / dHash), flags cross-label conflicts, stratified split, leakage check, data-quality report.
Feedback crops go to TRAIN only so the test set stays an untouched yardstick."""
import sys, json, yaml, shutil, random, hashlib, pathlib
import numpy as np
from PIL import Image
P = yaml.safe_load(open("params.yaml"))["data"]; lm = yaml.safe_load(open("configs/label_map.yaml")); random.seed(P["seed"])
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
def dhash(p):
    a = np.asarray(Image.open(p).convert("L").resize((9, 8), Image.LANCZOS), dtype=int); return int("".join("1" if v else "0" for v in (a[:, 1:] > a[:, :-1]).flatten()), 2)
rep = dict(corrupted=[], duplicates=[], label_conflicts=[], unmapped_dirs=[]); seen, by_cls = {}, {}
def add(f, tgt, train_only=False):
    try: Image.open(f).verify(); Image.open(f).load()
    except Exception: rep["corrupted"].append(str(f)); return
    keys = [("md5", md5(f)), ("dh", dhash(f))]
    for k in keys:
        if k in seen:
            rep["duplicates"].append(str(f))
            if seen[k][0] != tgt: rep["label_conflicts"].append(dict(file=str(f), label=tgt, other=seen[k][0], other_file=seen[k][1]))
            return
    for k in keys: seen[k] = (tgt, str(f))
    by_cls.setdefault(tgt, []).append((f, train_only))
for d in sorted(pathlib.Path("data/raw").glob("*/*")):
    if not d.is_dir(): continue
    tgt = lm.get(d.name)
    if not tgt: rep["unmapped_dirs"].append(str(d)); continue
    for f in sorted(d.glob("*.*")): add(f, tgt)
for d in sorted(pathlib.Path("data/feedback/crops").glob("*")):
    for f in sorted(d.glob("*.*")): add(f, d.name, True)
out = pathlib.Path("data/processed"); shutil.rmtree(out, ignore_errors=True); split_files = {"train": set(), "val": set(), "test": set()}; counts = {s: {} for s in split_files}
for tgt, items in by_cls.items():
    fb = [f for f, t in items if t]; base = [f for f, t in items if not t]; random.shuffle(base); n = len(base)
    nv, nt = (max(1, round(n * P["val"])), max(1, round(n * (1 - P["train"] - P["val"])))) if n >= 3 else (0, 0)
    for s, fs in (("val", base[:nv]), ("test", base[nv:nv + nt]), ("train", base[nv + nt:] + fb)):
        for f in fs:
            dst = out / s / tgt / f"{f.parent.parent.name}_{f.name}"; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, dst)
            split_files[s].add(md5(dst)); counts[s][tgt] = counts[s].get(tgt, 0) + 1
leak = sum(len(split_files[a] & split_files[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
tr = counts["train"]; rep.update(counts=counts, leakage=leak, imbalance_ratio=round(max(tr.values()) / max(min(tr.values()), 1), 2) if tr else None,
                                 classes_below_min=[c for c, n in tr.items() if n < P["min_per_class"]], n_corrupted=len(rep["corrupted"]), n_duplicates=len(rep["duplicates"]))
pathlib.Path("reports").mkdir(exist_ok=True); json.dump(rep, open("reports/data_quality.json", "w"), indent=2); print({k: v for k, v in rep.items() if not isinstance(v, (list, dict))})
if leak or not tr: sys.exit("data quality gate failed: leakage or empty dataset")
