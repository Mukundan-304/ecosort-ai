"""Integration check: loads the PROMOTED model through the same code path the app uses (src/ecosort/vision.py) and classifies a real crop."""
import sys, glob
sys.path.insert(0, "src")
import numpy as np
from PIL import Image
from ecosort import vision
from ecosort.knowledge import CLASSES
be = vision.backend(); assert be.startswith("classifier:"), f"app is NOT using the trained classifier (backend={be}); is models/classifier.pt + classes.json present and torch installed?"
f = sorted(glob.glob("data/processed/test/*/*.*"))[0]; img = Image.open(f).convert("RGB"); p = vision._classify(img)
assert p.shape == (len(CLASSES),) and abs(p.sum() - 1) < 1e-4; print("backend:", be, "| file:", f, "| top:", CLASSES[int(p.argmax())], f"{p.max():.1%}")
