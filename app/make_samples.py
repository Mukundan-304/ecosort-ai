import sys, numpy as np; sys.path.insert(0, "src")
from PIL import Image, ImageDraw
from ecosort.vision import PROTO
def make(path, seed=1):
    r = np.random.default_rng(seed); im = Image.new("RGB", (640, 420), (245, 245, 245)); d = ImageDraw.Draw(im)
    for i, c in enumerate(["plastic bottle", "glass bottle", "aluminum can", "cardboard box", "banana peel", "electronic device"]):
        x, y = 40 + (i % 3) * 200, 40 + (i // 3) * 190; col = tuple(int(v + r.integers(-8, 8)) for v in PROTO[c])
        (d.ellipse if i % 2 else d.rectangle)([x, y, x + 120, y + 120], fill=col)
    im.save(path)
if __name__ == "__main__": make("data/samples/sample1.png")
