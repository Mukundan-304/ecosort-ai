# ♻️ EcoSort AI — Waste-to-Recovery Orchestration Platform
Photo → per-item detection → material / recyclability / contamination / hazard → worker instructions (visual + voice) → **context-aware recovery routing** → uncertainty-gated human review → feedback → DVC retrain → redeploy. Every decision is written to a **hash-chained traceability ledger**.

## Architecture
```
Photo → Detector (YOLO | CV demo) → Material classifier (MobileNetV2 transfer learning)
      → Knowledge base (recyclable? hazard? handling) → Router (compat filter + distance/capacity/recovery/cost score)
      → Uncertainty gate (conf, entropy, hazard) ──low conf──► Human review ──► feedback.jsonl ─┐
      → Ledger (SHA-256 chain) → Analytics + Holt forecast                                      │
   DVC pipeline: prepare → train (MLflow) → evaluate (F1 quality gate) ◄─────────────────────────┘
   Jenkins: lint → test → dvc repro → docker build → scan → push → kubectl rollout (HPA, probes)
```
## Run
```bash
pip install -r requirements.txt && python app/make_samples.py
streamlit run app/streamlit_app.py      # demo backend works with zero downloads
docker compose up --build               # app :8501 + MLflow :5000
```
**Honest note:** without `models/material_clf.pt` the app uses a colour-prototype *demo* classifier so the full pipeline runs end-to-end. Train the real model (below) and it is picked up automatically.

## Data sources (download → `data/raw/<dataset>/<class>/*.jpg`, map classes in `configs/label_map.yaml`)
| Dataset | Use | Notes |
|---|---|---|
| **TACO** (tacodataset.org) | Multi-object detection/segmentation, litter "in the wild" | COCO format, ~60 classes; best for the multi-item detector |
| **ZeroWaste** (ai.bu.edu/zerowaste) | Detection/segmentation on conveyor-belt clutter | Real MRF, contamination-like scenes |
| **TrashNet** (GitHub garythung/trashnet) | Single-item classification baseline | ~2.5k imgs, 6 classes, clean backgrounds |
| **RealWaste** (UCI ML Repo) | Classification from a real landfill facility | ~4.7k imgs, 9 classes incl. food organics |
| **Kaggle "Garbage Classification" (12 classes)** | Larger classifier pool, adds batteries/clothes | Check licence & label noise |
| **Roboflow Universe** (search "waste detection") | Pre-annotated YOLO-format sets | Verify each licence |
| **Your own + human-review feedback** | Domain adaptation | The closed loop — this is your moat |

## Transfer learning (required — don't train from scratch)
- **Classifier:** ImageNet MobileNetV2 (or EfficientNet-B0). Phase 1 freeze backbone, train head; phase 2 unfreeze last blocks at lr/10. Augment heavily (crop, colour jitter) since waste is deformed/dirty.
- **Detector:** fine-tune YOLOv8n/s (COCO-pretrained) on TACO + ZeroWaste; set `ECOSORT_DETECTOR=yolo`.
- Combine datasets via label map; handle class imbalance (weighted loss / focal); calibrate probabilities (temperature scaling) before trusting the confidence gate.

## MLOps
```bash
pip install -r requirements-train.txt
dvc init && dvc remote add -d store s3://YOUR_BUCKET/ecosort   # or gdrive/local
dvc add data/raw && git add data/raw.dvc && dvc repro          # prepare → train → evaluate
dvc metrics show; mlflow ui
kubectl apply -f k8s/                                          # after setting image name
```
Jenkins needs credentials `dockerhub` and `kubeconfig`; replace `YOUR_DOCKERHUB_USER`.

## Resume bullets (fill in your real metrics!)
- Built an end-to-end waste-to-recovery platform: multi-object detection, transfer-learned material classifier (MobileNetV2), and a constraint-aware router assigning each item to a recovery facility by compatibility, distance, capacity and cost.
- Designed uncertainty-aware human-in-the-loop verification (confidence + entropy + hazard gate) whose corrections feed a DVC-versioned retraining pipeline with an F1 quality gate.
- Implemented SHA-256 hash-chained batch traceability and Holt-smoothing capacity forecasting.
- Productionised with Docker, Jenkins CI/CD (lint, test, dvc repro, scan, rollout), Kubernetes (HPA, probes), MLflow tracking.

## Interview talking points
- *Why hash-chain?* Tamper-evident audit without a blockchain's overhead.  *Why entropy + confidence?* Catches ambiguous-but-confident errors; hazards always reviewed.
- *Next steps:* temperature-scaled calibration, active learning (sample by entropy), drift monitoring (Evidently/Prometheus), FastAPI service split, real-time facility capacity via API, LP/OR-Tools batch routing.

## Real-photo mode (zero-shot AI)
`pip install -r requirements-ai.txt` → the app switches to **OWL-ViT** open-vocabulary detection (downloads ~600 MB once). It boxes each item in a real photo, names it (plastic bottle, banana peel, paper…), flags recyclable / biodegradable / hazard, and routes it to a facility. Without torch it falls back to the synthetic `demo` backend. Vocabulary lives in `src/ecosort/knowledge.py` — add items there.

## Classifier upgrade (transfer learning)
Pipeline: `dvc repro` = prepare (dedupe, leakage check) -> train (benchmark 3 backbones, pick by **validation** macro-F1) -> evaluate (held-out test, calibration, confusable-pair gate) -> promote (only if gate passes).
```bash
pip install -r requirements-train.txt
# put datasets in data/raw/<dataset>/<class>/*.jpg and edit configs/label_map.yaml
python training/prepare.py
python training/train.py                  # MLflow: ecosort-classifier  (mlflow ui)
python training/evaluate.py               # writes metrics.json, reports/confusion_matrix.png, reports/gate.json
python training/promote.py                # copies candidate -> models/ (loaded by the app)
python training/smoke_test.py             # confirms the app path uses the trained classifier
streamlit run app/streamlit_app.py
# or all at once: dvc repro
```
The sidebar shows `Detector: classifier:<arch>+<box source>` once the trained model is active. Optional `models/detector.pt` (YOLO) is used for boxes if present.

## Deploy (Docker -> Jenkins -> Kubernetes)
See the step list shared with this project: build image (`docker build -t ecosort-ai .`), run Jenkins from `jenkins/Dockerfile`, add credentials `dockerhub` + `kubeconfig`, create a Pipeline job from SCM, replace `YOUR_DOCKERHUB_USER` in `Jenkinsfile` and `k8s/ecosort.yaml`.

## AWS deployment
See `aws/README.md` (EKS cluster, ECR, Jenkins on EC2 via CloudFormation) and the `Jenkinsfile` (ECR + EKS). Local Docker Hub variant: `jenkins/Jenkinsfile.dockerhub`.
