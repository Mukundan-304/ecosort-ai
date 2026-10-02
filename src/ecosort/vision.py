"""Open-vocabulary waste detection. Backends: 'owl-vit' (zero-shot, real photos, needs torch+transformers) | 'demo' (classical CV on synthetic samples)."""
import os, json, threading, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from .knowledge import ITEMS, CLASSES

PROTO = {"plastic bottle": (150, 190, 215), "glass bottle": (60, 130, 80), "aluminum can": (170, 172, 178),
         "cardboard box": (175, 135, 85), "banana peel": (225, 200, 80), "electronic device": (40, 45, 60)}
_owl, _cache = None, {}

def _base_backend():
    if os.getenv("ECOSORT_BACKEND") == "demo": return "demo"
    try: import torch, transformers; return "owl-vit"
    except Exception: return "demo"

def _nms(b, s, thr=0.4):
    order, keep = np.argsort(-s), []
    while len(order):
        i, r = order[0], order[1:]; keep.append(i)
        if not len(r): break
        w = np.clip(np.minimum(b[i, 2], b[r, 2]) - np.maximum(b[i, 0], b[r, 0]), 0, None); h = np.clip(np.minimum(b[i, 3], b[r, 3]) - np.maximum(b[i, 1], b[r, 1]), 0, None)
        ar = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1]); inter = w * h
        order = r[inter / (ar(b[i]) + ar(b[r]) - inter + 1e-9) < thr]
    return keep

_raw, _cc = {}, {}

_lock = threading.Lock(); _status = {"state": "idle", "msg": ""}

def _device():
    import torch; return "cuda" if torch.cuda.is_available() else "cpu"

def _owl_load():
    global _owl
    with _lock:   # thread-safe: background warm-up and a user request never load the model twice
        if _owl is None:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            n = "google/owlvit-base-patch32"; _owl = (OwlViTProcessor.from_pretrained(n), OwlViTForObjectDetection.from_pretrained(n).eval().to(_device()))
    return _owl

def _owl_raw(img):
    """Heavy forward pass: runs ONCE per image. Detection-confidence / sensitivity sliders only re-filter these cached arrays."""
    key = hash(img.tobytes())
    if key not in _raw:
        import torch
        proc, model = _owl_load(); W, H = img.size
        inp = proc(text=[[f"a photo of {ITEMS[c]['prompt']}" for c in CLASSES]], images=img, return_tensors="pt")
        with torch.inference_mode(): o = model(**{k: v.to(_device()) for k, v in inp.items()})
        lg = o.logits[0].cpu().numpy(); cx, cy, w, h = o.pred_boxes[0].cpu().numpy().T
        box = np.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], 1).clip(0, [W, H, W, H])
        if len(_raw) > 8: _raw.clear()
        _raw[key] = (box, 1 / (1 + np.exp(-lg.max(1))), lg, W, H)
    return _raw[key]

def _owl_detect(img, det_thr, min_area):
    box, sc, lg, W, H = _owl_raw(img)
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]) / (W * H)
    idx = np.where((sc > det_thr) & (area > min_area) & (area < 0.9))[0]; out = []
    for k in (_nms(box[idx], sc[idx]) if len(idx) else []):
        i = idx[k]; e = np.exp(lg[i] - lg[i].max()); out.append((tuple(int(v) for v in box[i]), e / e.sum()))
    return out

def _demo_detect(img, min_area):
    a = np.asarray(img.convert("RGB")).astype(float); h, w, _ = a.shape
    bg = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), 0)
    lab, _ = ndi.label(ndi.binary_opening(np.linalg.norm(a - bg, axis=2) > 40, iterations=2)); out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        sel = lab[sl] == i
        if sel.sum() < min_area * h * w: continue
        d = np.array([np.linalg.norm(a[sl][sel].mean(0) - np.array(PROTO[c])) if c in PROTO else 1e3 for c in CLASSES]); e = np.exp(-d / 25)
        out.append(((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop), e / e.sum()))
    return out

ROOT = Path(os.getenv("ECOSORT_ROOT", Path(__file__).resolve().parents[2])); MODELS = ROOT / "models"; _clf = _meta = _yolo = None

def classifier_card():
    p = MODELS / "model_card.json"; return json.loads(p.read_text()) if p.exists() else None
def _clf_ready():
    if os.getenv("ECOSORT_BACKEND") == "demo" or not (MODELS / "classifier.pt").exists() or not (MODELS / "classes.json").exists(): return False
    try: import torch; return True
    except Exception: return False
def _yolo_ready():
    try: import ultralytics; return (MODELS / "detector.pt").exists()
    except Exception: return False
def backend():
    """classifier:<arch>+<box source> when the transfer-learned classifier is deployed; otherwise the previous zero-shot / demo backends."""
    if _clf_ready():
        return f"classifier:{(classifier_card() or {}).get('arch', 'custom')}+" + ("yolo" if _yolo_ready() else "owl-vit-boxes" if _base_backend() == "owl-vit" else "cv-boxes")
    return _base_backend()

def _boxes(img, det_thr, min_area):
    global _yolo
    W, H = img.size
    if _yolo_ready():
        from ultralytics import YOLO
        _yolo = _yolo or YOLO(str(MODELS / "detector.pt"))
        return [tuple(int(v) for v in b) for b in _yolo.predict(img, conf=det_thr, verbose=False)[0].boxes.xyxy.tolist() if (b[2] - b[0]) * (b[3] - b[1]) >= min_area * W * H]
    return [b for b, _ in (_owl_detect(img, det_thr, min_area) if _base_backend() == "owl-vit" else _demo_detect(img, min_area))]

def _classify(crop):
    """Transfer-learned classifier on an object crop -> calibrated probabilities over CLASSES."""
    global _clf, _meta
    import torch
    from .preprocess import eval_tf
    if _clf is None:
        cal = MODELS / "calibration.json"; card = classifier_card() or {}
        _clf = torch.jit.load(str(MODELS / "classifier.pt")).eval()
        _meta = dict(classes=json.loads((MODELS / "classes.json").read_text()), T=json.loads(cal.read_text())["temperature"] if cal.exists() else 1.0, size=card.get("img_size", 224))
    with torch.no_grad(): p = torch.softmax(_clf(eval_tf(_meta["size"])(crop)[None])[0] / _meta["T"], 0).numpy()
    full = np.zeros(len(CLASSES))
    for i, c in enumerate(_meta["classes"]):
        if c in CLASSES: full[CLASSES.index(c)] = p[i]
    if full.sum() == 0: raise ValueError(f"classifier classes {_meta['classes']} do not match knowledge-base classes; fix configs/label_map.yaml")
    return full / full.sum()

def warmup():
    """Load every model once and run a dummy pass, so the first real upload is fast. Returns the active backend name."""
    dummy = Image.new("RGB", (64, 64), (128, 128, 128))
    if _clf_ready(): _classify(dummy)
    if _base_backend() == "owl-vit": _owl_raw(dummy)
    return backend()

def detect_items(img, det_thr=0.12, min_area=0.002):
    """-> [(box_xyxy, probs over CLASSES)]. Image -> boxes (YOLO / OWL / CV) -> crop -> transfer-learned classifier (if deployed).
    Model forward passes and per-crop classifications are cached per image, so changing sliders/filters is instant."""
    be = backend(); ih = hash(img.tobytes()); key = (ih, round(det_thr, 3), round(min_area, 5), be)
    if key not in _cache:
        if len(_cache) > 32: _cache.clear()
        if be.startswith("classifier:"):
            W, H = img.size; res = []
            for b in _boxes(img, det_thr, min_area):
                if (ih, b) not in _cc:
                    if len(_cc) > 500: _cc.clear()
                    m = lambda: (max(0, b[0] - int(.04 * (b[2] - b[0]))), max(0, b[1] - int(.04 * (b[3] - b[1]))), min(W, b[2] + int(.04 * (b[2] - b[0]))), min(H, b[3] + int(.04 * (b[3] - b[1]))))
                    _cc[(ih, b)] = _classify(img.crop(m()))
                res.append((b, _cc[(ih, b)]))
            _cache[key] = res
        else: _cache[key] = _owl_detect(img, det_thr, min_area) if be == "owl-vit" else _demo_detect(img, min_area)
    return _cache[key]

def warmup_async():
    """Start loading models in a background thread (idempotent per process) so the UI renders immediately instead of blocking."""
    if _status["state"] != "idle": return
    _status.update(state="loading", msg="")
    def run():
        try: warmup(); _status.update(state="ready")
        except Exception as e: _status.update(state="error", msg=f"{type(e).__name__}: {e}")
    threading.Thread(target=run, daemon=True, name="ecosort-warmup").start()
def model_status(): return dict(_status)
