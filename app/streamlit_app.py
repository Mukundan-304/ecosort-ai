import sys, os, json, io, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)  # work from any launch folder
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "app")]
SAMPLE = os.path.join("data", "samples", "sample1.png")
if not os.path.exists(SAMPLE):  # auto-create demo image if missing
    os.makedirs(os.path.dirname(SAMPLE), exist_ok=True); from make_samples import make; make(SAMPLE)
import streamlit as st, pandas as pd, streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont
from ecosort.pipeline import analyze
from ecosort.ops import analytics, forecast, save_feedback, ledger_verify, ledger_append, LEDGER
from ecosort.knowledge import CLASSES
from ecosort.vision import backend, warmup_async, model_status

STREAM_COL = {"plastic_pet": "#1f77b4", "plastic_hdpe": "#17becf", "glass": "#2ca02c", "metal": "#7f7f7f", "paper": "#ff7f0e", "organic": "#8c564b", "ewaste": "#d62728", "residual": "#9467bd", "textile": "#e377c2"}

def badges(it):
    return " ".join([("♻️ recyclable" if it["recyclable"] else "🚫 not recyclable"), ("🌱 biodegradable" if it["biodegradable"] else ""), ("☣️ HAZARD" if it["hazard"] else "")])
st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="wide")

def annotate(img, items, sel=None):
    a = img.copy(); d = ImageDraw.Draw(a); f = ImageFont.load_default(size=15)
    for i, it in enumerate(items):
        x0, y0, x1, y1 = it["box"]; c = STREAM_COL.get(it["stream"], "#999999")
        d.rectangle([x0, y0, x1, y1], outline=c, width=7 if i == sel else 3)
        if it["review"]: d.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline="red", width=1)
        tag = f"#{i} {it['material']} {it['conf']:.0%}" + (" !" if it["review"] else "")
        tb = d.textbbox((0, 0), tag, font=f); tw, th = tb[2] - tb[0] + 10, tb[3] - tb[1] + 8
        ty = y0 - th if y0 > th else y0
        d.rectangle([x0, ty, x0 + tw, ty + th], fill=c); d.text((x0 + 5, ty + 3), tag, fill="white", font=f)
    return a

st.title("♻️ EcoSort AI — Waste-to-Recovery Orchestrator")

@st.cache_data(show_spinner=False, max_entries=16)
def run_analysis(raw: bytes, lat, lon, thr, min_area, det_thr):
    im = Image.open(io.BytesIO(raw)).convert("RGB"); im.thumbnail((1024, 1024))
    t = time.perf_counter(); b, it = analyze(im, (lat, lon), thr, log=False, min_area=min_area, det_thr=det_thr); return b, it, (time.perf_counter() - t) * 1000

warmup_async()   # loads models in a background thread; the page renders immediately (no blocking banner)
with st.sidebar:
    st.header("Input"); mode = st.radio("Image source", ["Sample", "Upload", "Camera"], horizontal=True)
    file = (st.file_uploader("Waste photo", type=["png", "jpg", "jpeg"]) if mode == "Upload"
            else st.camera_input("Take a photo") if mode == "Camera" else SAMPLE)
    st.header("Site context"); lat = st.number_input("Latitude", value=12.95); lon = st.number_input("Longitude", value=77.60)
    st.header("Tuning"); thr = st.slider("Review confidence threshold", 0.3, 0.95, 0.6)
    sens = st.slider("Detection sensitivity", 1, 10, 5, help="Higher = detects smaller objects"); dthr = st.slider("Detection confidence", 0.05, 0.5, 0.12, help="Lower = more boxes (real-photo AI mode)")
    st.header("Filters"); only_rev = st.toggle("Only items needing review"); 
st.sidebar.caption(f"Detector: **{backend()}**" + (" — install torch+transformers for real photos" if backend() == "demo" else ""))
_ms = model_status()
st.sidebar.caption({"loading": "⏳ AI model warming up in background…", "ready": "✅ AI model ready", "error": "❌ Model load failed: " + _ms["msg"], "idle": ""}[_ms["state"]])
if not file:
    st.info("Upload a photo to start." + (" The AI model is warming up in the background." if _ms["state"] == "loading" else ""))
    if _ms["state"] == "loading": time.sleep(1.5); st.rerun()
    st.stop()

img = Image.open(file).convert("RGB"); img.thumbnail((1024, 1024)); fname = getattr(file, "name", "sample1.png")
raw = file.getvalue() if hasattr(file, "getvalue") else open(file, "rb").read()
try:
    with st.spinner("Waiting for the model to finish loading (first launch only)…" if model_status()["state"] == "loading" else "Detecting and classifying items…"):
        batch, items, ms = run_analysis(raw, lat, lon, thr, 0.02 / sens ** 1.5 / 3, dthr)   # slow only for a NEW image; sliders/filters reuse cached output
except Exception as e: st.error(f"Inference failed: {type(e).__name__}: {e}"); st.stop()
mats = st.sidebar.multiselect("Materials", CLASSES, default=CLASSES)
vis = [(i, it) for i, it in enumerate(items) if it["material"] in mats and (it["review"] or not only_rev)]
t1, t2, t3, t4 = st.tabs(["🔍 Analyze", "🧑‍🔧 Human review", "📈 Analytics & forecast", "🔗 Traceability"])

with t1:
    if not items: st.warning("No objects detected — raise detection sensitivity."); st.stop()
    a = analytics(items); m = st.columns(5)
    m[0].metric("Items detected", a["items"]); m[1].metric("Diversion %", a["diversion_pct"]); m[2].metric("Recovered kg", a["recovered_kg"])
    m[3].metric("Needs review", f"{sum(i['review'] for i in items)}/{len(items)}"); m[4].metric("Batch", batch); st.caption(f"Analysis time {ms / 1000:.2f} s · {backend()}")
    L, R = st.columns([1.3, 1])
    sel = R.selectbox("🔎 Inspect item (highlights its box)", [None] + [i for i, _ in vis], format_func=lambda i: "— none —" if i is None else f"#{i} {items[i]['material']}")
    L.image(annotate(img, [it for _, it in enumerate(items)], sel), caption="Coloured box = material · red outline = needs human review", use_container_width=True)
    if sel is not None:
        it = items[sel]
        with R.container(border=True):
            c = st.columns([1, 1.5]); c[0].image(img.crop(it["box"]), use_container_width=True)
            c[1].subheader(f"{it['icon']} {it['material']}"); c[1].progress(min(it["conf"], 1.0), text=f"Confidence {it['conf']:.0%} · entropy {it['entropy']:.2f}")
            c[1].write(f"**{it['kind']}**  \n{badges(it)}")
            st.info(it["instruction"]); st.write(f"🚚 **{it['destination']}** ({it['km']} km, {it['recovery']:.0%} recovery)"); st.caption("Alternatives: " + (", ".join(it["alternatives"]) or "none"))
    else: R.bar_chart(pd.Series([it["material"] for it in items]).value_counts())
    st.subheader(f"Detected items ({len(vis)})")
    for r in range(0, len(vis), 3):
        for col, (i, it) in zip(st.columns(3), vis[r:r + 3]):
            with col.container(border=True):
                c = st.columns([1, 1.6]); c[0].image(img.crop(it["box"]), use_container_width=True)
                c[1].markdown(f"**#{i} {it['icon']} {it['material']}**  \n{'🟥 REVIEW' if it['review'] else '🟩 auto'} · {it['conf']:.0%}  \n{badges(it)}")
                c[1].caption(f"➜ {it['destination']}")
                with st.expander("Instruction"): st.write(it["instruction"])
    st.subheader("🚚 Dispatch plan")
    st.dataframe(pd.DataFrame(items).groupby("destination").agg(items=("material", "count"), what=("material", lambda x: ", ".join(sorted(set(x)))), est_kg=("kg", "sum"), km=("km", "first")).reset_index(), use_container_width=True)
    b = st.columns([1, 1, 3])
    with b[0]:
        speech = ". ".join(f"Item {i}: {it['material']}. {it['instruction']}" for i, it in enumerate(items))
        components.html(f"<button style='padding:8px 14px;font-size:15px' onclick=\"speechSynthesis.cancel();speechSynthesis.speak(new SpeechSynthesisUtterance({json.dumps(speech)}))\">🔊 Voice instructions</button>", height=50)
    if b[1].button("✅ Commit batch to ledger"):
        for it in items: ledger_append(dict(batch=batch, material=it["material"], dest=it["destination"], kg=it["kg"], review=it["review"]))
        st.toast(f"Batch {batch}: {len(items)} records written", icon="🔗")

with t2:
    q = [(i, it) for i, it in enumerate(items) if it["review"]]; st.write(f"**{len(q)}** item(s) flagged (low confidence / high entropy / hazardous).")
    for i, it in q:
        with st.container(border=True):
            c = st.columns([1, 2, 2, 1]); c[0].image(img.crop(it["box"]), use_container_width=True)
            c[1].write(f"#{i} predicted **{it['material']}** ({it['conf']:.0%})")
            fix = c[2].selectbox("Correct label", CLASSES, index=CLASSES.index(it["material"]), key=f"f{batch}{i}")
            if c[3].button("Confirm", key=f"b{batch}{i}"): save_feedback(fname, it["box"], it["material"], fix); st.toast("Saved → feeds next DVC retrain", icon="🧠")

with t3:
    h, f = forecast(); st.line_chart(pd.concat([h, f.set_axis(range(len(h), len(h) + len(f)))], axis=1)); st.caption("Holt-smoothing 7-day waste forecast → capacity planning")

with t4:
    st.write("Ledger integrity:", "✅ valid" if ledger_verify() else "❌ tampered")
    if os.path.exists(LEDGER): st.dataframe(pd.read_json(LEDGER, lines=True).tail(20), use_container_width=True)
    else: st.caption("No batches committed yet.")
