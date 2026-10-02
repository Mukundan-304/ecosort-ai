"""Context-aware routing: hard filter on material compatibility/capacity, soft score on distance, contamination, cost, recovery value."""
import math, yaml, os
def _hav(a, b):
    p = math.pi / 180; dl = (b[1] - a[1]) * p; d = math.sin((b[0]-a[0])*p/2)**2 + math.cos(a[0]*p)*math.cos(b[0]*p)*math.sin(dl/2)**2
    return 12742 * math.asin(math.sqrt(d))
def facilities(path=None):
    return yaml.safe_load(open(path or os.getenv("ECOSORT_FACILITIES", "configs/facilities.yaml")))["facilities"]
def route(stream, site, kg, contaminated=False, fac=None, w=(0.4, 0.25, 0.2, 0.15)):
    cands = []
    for f in fac or facilities():
        if stream not in f["accepts"] or f["capacity_kg"] - f["load_kg"] < kg: continue
        dist = _hav(site, (f["lat"], f["lon"]))
        s = (w[0] * math.exp(-dist / 15) + w[1] * (1 - f["load_kg"] / f["capacity_kg"])
             + w[2] * f["recovery_rate"] + w[3] * (0.3 if contaminated and f["recovery_rate"] > 0.8 else 1) * (1 - f["cost_per_kg"] / 10))
        cands.append(dict(facility=f["name"], km=round(dist, 1), score=round(s, 3), recovery=f["recovery_rate"]))
    cands.sort(key=lambda c: -c["score"])
    return cands[0] if cands else dict(facility="Landfill/RDF (no compatible facility)", km=None, score=0, recovery=0), cands[1:3]
