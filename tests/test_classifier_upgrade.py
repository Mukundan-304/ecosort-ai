import sys, json, subprocess, pathlib, numpy as np
from PIL import Image
sys.path.insert(0, "src")
from ecosort.evalutils import *
from ecosort.preprocess import pad_square
def test_class_weights_favor_rare():
    w = class_weights([1000, 100, 10]); assert w[2] > w[1] > w[0] and abs(w.sum() - 3) < 1e-9
def test_temperature_fixes_overconfidence():
    r = np.random.default_rng(0); y = r.integers(0, 4, 3000); z = r.normal(0, 3, (3000, 4)); wrong = r.random(3000) < .3
    z[np.arange(3000), np.where(wrong, (y + 1) % 4, y)] += 8; T = fit_temperature(z, y)
    assert T > 1.2 and nll(z, y, T) < nll(z, y, 1) and ece(softmax(z, T), y) < ece(softmax(z), y)
def test_f1_and_pairs():
    y = np.array([0] * 40 + [1] * 40); pred = np.array([0] * 30 + [1] * 10 + [1] * 40); cm = confusion(y, pred, 2)
    assert abs(macro_f1(y, pred, 2) - np.nanmean(f1_per_class(cm))) < 1e-9
    pr = pair_rates(cm, ["paper", "aluminum can"], [dict(name="p-m", a=["paper"], b=["aluminum can"])], 20); assert pr[0]["evaluated"] and abs(pr[0]["rate"] - 0.25) < 1e-9 and pr[1]["rate"] == 0.0
def test_pad_square_keeps_aspect():
    o = pad_square(Image.new("RGB", (100, 40), (1, 2, 3))); assert o.size == (100, 100) and o.getpixel((50, 50)) == (1, 2, 3)
def test_prepare_dedup_and_no_leakage(tmp_path):
    import shutil; root = pathlib.Path.cwd(); shutil.copytree("configs", tmp_path / "configs"); shutil.copy("params.yaml", tmp_path / "params.yaml"); r = np.random.default_rng(1)
    for cls in ("paper", "glass"):
        d = tmp_path / "data/raw/ds" / cls; d.mkdir(parents=True)
        for i in range(30): Image.fromarray(r.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(d / f"{i}.png")
        shutil.copy(d / "0.png", d / "dup.png")
    (tmp_path / "data/raw/ds/paper/bad.png").write_bytes(b"not an image")
    subprocess.run([sys.executable, str(root / "training/prepare.py")], cwd=tmp_path, check=True)
    rep = json.load(open(tmp_path / "reports/data_quality.json")); assert rep["n_corrupted"] == 1 and rep["n_duplicates"] >= 2 and rep["leakage"] == 0
