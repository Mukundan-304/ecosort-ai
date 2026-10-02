import numpy as np, uuid
from .vision import detect_items
from .knowledge import ITEMS, CLASSES
from .routing import route
from .ops import review_needed, ledger_append

def analyze(img, site=(12.95, 77.6), thr=0.6, log=True, min_area=0.002, det_thr=0.12):
    batch, items = "B-" + uuid.uuid4().hex[:8], []
    for box, p in detect_items(img, det_thr, min_area):
        k = int(p.argmax()); name = CLASSES[k]; kb = ITEMS[name]
        ent = float(-(p * np.log(p + 1e-9)).sum() / np.log(len(p)))
        dest, alts = route(kb["stream"], site, kb["kg"])
        it = dict(box=box, material=name, kind=kb["kind"], stream=kb["stream"], conf=round(float(p[k]), 3), entropy=round(ent, 3),
                  recyclable=kb["recyclable"], biodegradable=kb["biodegradable"], hazard=kb["hazard"], instruction=kb["tip"], icon=kb["icon"], kg=kb["kg"],
                  destination=dest["facility"], km=dest["km"], recovery=dest["recovery"], review=review_needed(p[k], ent, kb["hazard"], thr),
                  alternatives=[a["facility"] for a in alts])
        items.append(it)
        if log: ledger_append(dict(batch=batch, material=name, dest=dest["facility"], kg=kb["kg"], review=it["review"]))
    return batch, items
