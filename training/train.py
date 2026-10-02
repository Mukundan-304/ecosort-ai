"""Transfer-learning benchmark. For each of MobileNetV3-Large / EfficientNetV2-S / ConvNeXt-Tiny:
 ImageNet weights -> new head -> (1) freeze backbone, train head -> (2) hard-negative mining -> (3) unfreeze last blocks, fine-tune at small LR
 (augmentation, class balancing, label smoothing, ReduceLROnPlateau, early stopping, best-checkpoint) -> temperature calibration on VAL.
Model choice is made on VALIDATION macro-F1 only (test set is untouched until evaluate.py). Winner exported as TorchScript to models/candidate/."""
import sys, json, time, copy, hashlib, pathlib, subprocess, datetime as dt, os, yaml, numpy as np, torch, torch.nn as nn, mlflow
sys.path.insert(0, "src")
from torchvision import datasets, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from ecosort.preprocess import eval_tf, train_tf
from ecosort.evalutils import class_weights, fit_temperature, ece, softmax, macro_f1, weighted_f1, accuracy, nll
P = yaml.safe_load(open("params.yaml"))["train"]; dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(P["seed"]); np.random.seed(P["seed"]); S = P["img_size"]; cand = pathlib.Path("models/candidate"); cand.mkdir(parents=True, exist_ok=True)
tr = datasets.ImageFolder("data/processed/train", train_tf(S)); tr_eval = datasets.ImageFolder("data/processed/train", eval_tf(S)); va = datasets.ImageFolder("data/processed/val", eval_tf(S))
classes, n = tr.classes, len(tr.classes); assert va.classes == classes, "train/val class sets differ"
ytr = np.array(tr.targets); counts = np.bincount(ytr, minlength=n); cw = class_weights(counts)
mk = lambda ds, **kw: DataLoader(ds, P["batch"], num_workers=P["num_workers"], **kw); VL = mk(va); TEL = mk(tr_eval)

def build(arch):
    if arch == "mobilenet_v3_large": m = models.mobilenet_v3_large(weights="DEFAULT"); m.classifier[3] = nn.Linear(m.classifier[3].in_features, n)
    elif arch == "efficientnet_v2_s": m = models.efficientnet_v2_s(weights="DEFAULT"); m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    elif arch == "convnext_tiny": m = models.convnext_tiny(weights="DEFAULT"); m.classifier[2] = nn.Linear(m.classifier[2].in_features, n)
    else: raise ValueError(arch)
    return m.to(dev)
def logits(m, loader):
    m.eval(); zs, ys = [], []
    with torch.no_grad():
        for x, y in loader: zs.append(m(x.to(dev)).float().cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(zs), np.concatenate(ys)
def fit(m, groups, epochs, loader, crit, k, tag, ck):
    opt = torch.optim.AdamW(groups, weight_decay=1e-4); sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=1); best, bad, sd = -1, 0, None
    for e in range(epochs):
        m.train(); m.features[: len(m.features) - k].eval()  # frozen blocks keep BatchNorm statistics fixed
        for x, y in loader: opt.zero_grad(); crit(m(x.to(dev)), y.to(dev)).backward(); opt.step()
        z, y = logits(m, VL); f1 = macro_f1(y, z.argmax(1), n); sch.step(f1); mlflow.log_metrics({f"{tag}_val_macro_f1": f1, f"{tag}_val_loss": nll(z, y)}, step=e)
        if f1 > best: best, bad, sd = f1, 0, copy.deepcopy(m.state_dict()); torch.save(sd, ck)
        else:
            bad += 1
            if bad >= P["patience"]: break
    m.load_state_dict(sd); return best
def latency_ms(m):
    mc = copy.deepcopy(m).cpu().eval(); x = torch.randn(1, 3, S, S); ts = []
    with torch.no_grad():
        for _ in range(35): t = time.perf_counter(); mc(x); ts.append(time.perf_counter() - t)
    return float(np.median(ts[5:]) * 1000)

def train_arch(arch):
    with mlflow.start_run(run_name=arch, nested=True) as run:
        mlflow.log_params({**{k: v for k, v in P.items() if not isinstance(v, dict)}, "arch": arch, "unfreeze_blocks": P["unfreeze_blocks"][arch], "lr_ft": P["lr_ft"][arch], "n_classes": n, "train_counts": json.dumps(dict(zip(classes, counts.tolist())))})
        m = build(arch); k = P["unfreeze_blocks"][arch]; ck = cand / f"{arch}_best.pt"
        for p in m.features.parameters(): p.requires_grad = False
        f1_head = fit(m, [{"params": m.classifier.parameters(), "lr": P["lr_head"]}], P["head_epochs"], mk(tr, shuffle=True, drop_last=True), nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=dev), label_smoothing=P["label_smoothing"]), 0, "head", ck)
        z, y = logits(m, TEL); hard = z.argmax(1) != y; w = cw[y] * np.where(hard, P["hard_neg_boost"], 1.0)  # hard-negative mining: confused train samples are sampled more often
        mlflow.log_metrics({"hard_negatives": int(hard.sum()), "head_val_macro_f1_best": f1_head})
        for p in m.features[-k:].parameters(): p.requires_grad = True
        groups = [{"params": list(m.features[-k:].parameters()), "lr": P["lr_ft"][arch]}, {"params": m.classifier.parameters(), "lr": P["lr_ft"][arch] * 5}]
        sampler = WeightedRandomSampler(w, len(w), replacement=True)
        f1_ft = fit(m, groups, P["ft_epochs"], mk(tr, sampler=sampler), nn.CrossEntropyLoss(label_smoothing=P["label_smoothing"]), k, "finetune", ck)
        z, y = logits(m, VL); T = fit_temperature(z, y); pr = softmax(z, T)
        r = dict(arch=arch, run_id=run.info.run_id, val_macro_f1=macro_f1(y, z.argmax(1), n), val_weighted_f1=weighted_f1(y, z.argmax(1), n), val_accuracy=accuracy(y, z.argmax(1)),
                 val_ece_uncalibrated=ece(softmax(z), y), val_ece_calibrated=ece(pr, y), temperature=T, latency_ms=latency_ms(m), params_m=sum(p.numel() for p in m.parameters()) / 1e6, hard_negatives=int(hard.sum()))
        mlflow.log_metrics({k_: v for k_, v in r.items() if isinstance(v, float)}); print(json.dumps(r)); return r

if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")); mlflow.set_experiment("ecosort-classifier")
    with mlflow.start_run(run_name="benchmark"):
        results = [train_arch(a) for a in P["archs"]]
        top = max(r["val_macro_f1"] for r in results); chosen = min([r for r in results if r["val_macro_f1"] >= top - P["tie_margin"]], key=lambda r: r["latency_ms"])
    m = build(chosen["arch"]); m.load_state_dict(torch.load(cand / f"{chosen['arch']}_best.pt", map_location=dev)); m.cpu().eval()
    torch.jit.trace(m, torch.randn(1, 3, S, S), check_trace=False).save(str(cand / "classifier.pt"))
    json.dump(classes, open(cand / "classes.json", "w")); json.dump({"temperature": chosen["temperature"]}, open(cand / "calibration.json", "w"))
    h = hashlib.sha1()
    for p in sorted(pathlib.Path("data/processed").rglob("*.*")): h.update(f"{p.relative_to('data/processed')}:{p.stat().st_size}".encode())
    try: rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception: rev = None
    card = dict(version=f"ecosort-clf-{chosen['arch']}-{dt.date.today():%Y%m%d}", arch=chosen["arch"], img_size=S, classes=classes, train_class_distribution=dict(zip(classes, counts.tolist())), temperature=chosen["temperature"],
                val_macro_f1=chosen["val_macro_f1"], val_weighted_f1=chosen["val_weighted_f1"], val_accuracy=chosen["val_accuracy"], latency_ms=chosen["latency_ms"], dataset_version=h.hexdigest()[:12],
                git_rev=rev, mlflow_run_id=chosen["run_id"], trained_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    json.dump(card, open(cand / "model_card.json", "w"), indent=2); json.dump(dict(chosen=chosen["arch"], selection="highest validation macro-F1; ties within tie_margin resolved by lower CPU latency", results=results, dataset_version=card["dataset_version"]), open("reports/benchmark.json", "w"), indent=2)
    with mlflow.start_run(run_id=chosen["run_id"]): mlflow.log_artifact(str(cand / "classifier.pt")); mlflow.log_artifact(str(cand / "model_card.json"))
    print("chosen:", chosen["arch"])
