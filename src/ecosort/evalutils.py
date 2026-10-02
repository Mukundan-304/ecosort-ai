"""Numpy-only metrics: class balancing, temperature scaling, ECE, F1, confusion, confusable-pair rates."""
import numpy as np
def class_weights(counts, beta=0.999):
    counts = np.asarray(counts, float); eff = (1 - beta ** counts) / (1 - beta); w = 1 / np.maximum(eff, 1e-9); return w / w.sum() * len(counts)
def softmax(z, T=1.0):
    z = np.asarray(z, float) / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def nll(z, y, T=1.0): return float(-np.log(softmax(z, T)[np.arange(len(y)), y] + 1e-12).mean())
def fit_temperature(z, y, grid=np.linspace(0.5, 5.0, 91)): return float(min(grid, key=lambda T: nll(z, y, T)))
def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1); acc = (pred == y).astype(float); e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)
def confusion(y, pred, n):
    cm = np.zeros((n, n), int); np.add.at(cm, (np.asarray(y), np.asarray(pred)), 1); return cm
def f1_per_class(cm):
    tp = np.diag(cm).astype(float); d = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp); return np.where(d > 0, 2 * tp / np.maximum(d, 1), np.nan)
def macro_f1(y, pred, n): return float(np.nanmean(f1_per_class(confusion(y, pred, n))))
def weighted_f1(y, pred, n):
    cm = confusion(y, pred, n); f = f1_per_class(cm); s = cm.sum(1); m = ~np.isnan(f); return float((f[m] * s[m]).sum() / s[m].sum())
def accuracy(y, pred): return float((np.asarray(y) == np.asarray(pred)).mean())
def pair_rates(cm, classes, pairs, min_support=20):
    """Directional confusion rate between visually-similar groups, e.g. true paper predicted as metal."""
    idx = {c: i for i, c in enumerate(classes)}; out = []
    for p in pairs:
        A = [idx[c] for c in p["a"] if c in idx]; B = [idx[c] for c in p["b"] if c in idx]
        for name, S, D in ((f"{p['name']} a->b", A, B), (f"{p['name']} b->a", B, A)):
            sup = int(cm[S, :].sum()) if S else 0
            out.append(dict(pair=name, support=sup, evaluated=bool(S and D and sup >= min_support), rate=float(cm[np.ix_(S, D)].sum() / sup) if S and D and sup >= min_support else None))
    return out
