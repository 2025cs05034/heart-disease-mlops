# heart-disease-mlops

MLOps assignment (BITS Pilani, M.Tech, S2-25_AMLCSZG523).

Trains a heart-disease classifier on the UCI Cleveland dataset (303 records),
serves it through FastAPI inside a Docker container, and runs it on Kubernetes
with Prometheus + Grafana for monitoring.

Best model: **Logistic Regression**, test ROC-AUC `0.9588`,
test accuracy `0.8689`.

PNG exports of the architecture diagram are in
`screenshots/architecture_diagram_1.png` and
`screenshots/architecture_diagram_2.png`.

---

## Prerequisites

Tested on macOS (Apple Silicon) and Ubuntu 22.04. Should also work on Windows
via WSL2.

| Tool | Version | Required for |
| --- | --- | --- |
| Python | 3.11 | training, tests, API |
| Git | any recent | cloning the repo |
| Docker Desktop / Docker Engine | 20+ | containerised API, monitoring stack |
| kubectl | 1.27+ | Kubernetes deployment |
| helm | 3.12+ | Kubernetes deployment via Helm chart |
| make | any | optional, wraps the common commands |

Install hints:

- **macOS (Homebrew):** `brew install python@3.11 git kubectl helm` — Docker
  Desktop comes with `docker` and `docker compose` built in.
- **Ubuntu/Debian:** `sudo apt install python3.11 python3.11-venv git make`,
  then install Docker Engine, kubectl, and Helm from their official repos.
- **Windows:** use WSL2 (Ubuntu 22.04) and follow the Ubuntu instructions.

The Kubernetes section assumes a local single-node cluster. Two easy options:

1. **Docker Desktop:** Settings → Kubernetes → "Enable Kubernetes".
2. **kind** (`brew install kind` then `kind create cluster`).

---

## Quickstart

The fastest way to verify everything works end-to-end (target time: ~5 minutes).

```bash
git clone https://github.com/2025cs05034/heart-disease-mlops.git
cd heart-disease-mlops
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make data        # downloads UCI dataset
make train       # trains LR + RF, writes models/best_model.pkl
make test        # 23 tests should pass
make api-run &   # starts API on http://localhost:8000

# in another shell
curl http://localhost:8000/health        # {"status":"ok"}
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":0,"trestbps":160,"chol":286,"fbs":0,"restecg":0,
       "thalach":108,"exang":1,"oldpeak":1.5,"slope":1,"ca":3,"thal":3}'
# {"prediction":1,"label":"heart_disease",...}
```

If those commands all succeed, the core pipeline is working. The rest of the
README expands each step and adds the Docker / Kubernetes / monitoring paths.

If `make` is not installed, every target maps to a plain command shown in the
relevant section below.

---

## Setup

```bash
git clone https://github.com/2025cs05034/heart-disease-mlops.git
cd heart-disease-mlops
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`make help` lists every available target with a one-line description.

---

## Train

```bash
python -m src.download_data     # writes data/raw/heart_disease.csv
python -m src.train             # trains LR + RF, picks the best
```

Or with `make`:

```bash
make data
make train
```

What to expect:

- `data/raw/heart_disease.csv` is created (303 rows, 14 columns).
- Training runs five-fold stratified cross-validation with `GridSearchCV` on
  both Logistic Regression and Random Forest, scoring on ROC-AUC.
- The best pipeline (preprocessor + estimator) is saved to
  `models/best_model.pkl` together with `models/best_model_metadata.json`.
- Plots and a summary are written to `artifacts/`.
- Each model run is logged to MLflow with a SQLite backend (`mlflow.db`).

To browse the MLflow runs:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5050
# or
make mlflow-ui
```

Then open <http://localhost:5050>. Port 5050 is used because macOS reserves
port 5000 for AirPlay; on Linux 5000 also works.

---

## Tests and lint

```bash
pytest tests/                     # 23 tests
flake8 src tests
black --check src tests
isort --check-only src tests
```

Or:

```bash
make test
make lint
```

CI runs the same four steps plus a Docker build smoke test on every push to
`main`; see `.github/workflows/ci.yml`.

---

## API

The API requires a trained model on disk (`models/best_model.pkl`), so run
`make train` once before this step.

```bash
uvicorn src.api:app --port 8000
# or
make api-run
```

Endpoints:

- `GET /health` — liveness probe; returns `{"status":"ok"}`.
- `GET /ready` — readiness probe; returns `{"status":"ready","model_loaded":true}`.
- `POST /predict` — single prediction (see sample below).
- `POST /predict/batch` — many predictions in one call.
- `GET /metrics` — Prometheus-format metrics.

Sample call:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":0,"trestbps":160,"chol":286,"fbs":0,"restecg":0,
       "thalach":108,"exang":1,"oldpeak":1.5,"slope":1,"ca":3,"thal":3}'
```

Response includes `prediction`, `label`, `confidence`, `probability_of_disease`,
`model_version`, and a per-request `request_id` that is also returned in the
`X-Request-Id` response header for log correlation.

---

## Docker

The Dockerfile copies the trained model into the image, so training must
have produced `models/best_model.pkl` before the build.

```bash
make train                                    # if not already done
docker build -t heart-disease-api:1.0.0 .
docker run --rm -p 8000:8000 heart-disease-api:1.0.0

# verify
curl http://localhost:8000/health
```

The image runs as UID 1001 (non-root) and ships with a Python healthcheck.
Final image size is around 200–250 MB.

---

## Kubernetes

Two options ship with the repo. Either works; pick one.

### Prerequisite

Make sure a local Kubernetes cluster is running and `kubectl` points to it.

```bash
kubectl cluster-info       # should return cluster URLs
kubectl get nodes          # should list at least one Ready node
```

If the image was built locally, it must be available to the cluster. With
Docker Desktop's built-in Kubernetes the local image is shared automatically.
With `kind`, load the image first: `kind load docker-image heart-disease-api:1.0.0`.

### Option A — raw manifests

```bash
kubectl apply -f deploy/k8s/
kubectl get all -n heart-disease
```

### Option B — Helm

```bash
helm install heart-disease-api deploy/helm/heart-disease-api \
     --namespace heart-disease --create-namespace
helm status heart-disease-api -n heart-disease
```

Both options ship a Deployment, Service, Ingress, an HPA (2–5 replicas, 70%
CPU target) and a default-deny NetworkPolicy.

### Hitting the API on the cluster

The simplest way to test the running pods is `port-forward` (no Ingress
controller required):

```bash
kubectl -n heart-disease port-forward svc/heart-disease-api 8000:80
curl http://localhost:8000/health
```

If an Ingress controller (e.g. `ingress-nginx`) is installed and the Ingress
host is mapped in `/etc/hosts`, requests can also go through the Ingress.

---

## Monitoring

Local stack via Docker Compose (Prometheus + Grafana):

```bash
cp deploy/monitoring/grafana_admin_password.txt.example \
   deploy/monitoring/grafana_admin_password.txt
# edit the file and set an admin password, then:
make monitoring-up
# or:
docker compose -f deploy/monitoring/docker-compose.monitoring.yml up -d
```

- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> (login with `admin` / the password set above)

A Grafana dashboard called **"Heart Disease API"** is auto-provisioned from
`deploy/monitoring/grafana/dashboards/`. It plots request rate, latency
percentiles, total predictions, and disease vs no-disease counts.

**Important — scrape target:** Prometheus is configured to scrape the API at
`host.docker.internal:8000`. This works when the API is running on the host
(via `make api-run` or `docker run -p 8000:8000 ...` on the same machine that
runs Docker Desktop). If the API runs only inside Kubernetes, edit
`deploy/monitoring/prometheus/prometheus.yml` to point at the cluster's
service address before starting the stack.

---

## Repo layout

```
src/                training, inference, FastAPI service
tests/              pytest suite (23 tests)
notebooks/          EDA notebook (executed, with outputs preserved)
deploy/k8s/         raw Kubernetes manifests
deploy/helm/        Helm chart
deploy/monitoring/  Prometheus + Grafana docker-compose stack
.github/workflows/  GitHub Actions CI
screenshots/        evidence images used in the report
artifacts/          training plots and summary (created by `make train`)
models/             best_model.pkl + metadata (created by `make train`)
```

---

## Dataset

UCI Heart Disease, Cleveland subset.
<https://archive.ics.uci.edu/dataset/45/heart+disease>

`src/download_data.py` fetches the raw CSV, falls back to the `ucimlrepo`
package if the direct URL fails, drops malformed rows, coerces feature
columns to numeric, and collapses the 0–4 severity target into a binary
0/1 label.

---

## Troubleshooting

- **`pip install` fails on `numpy` / `scipy`:** make sure Python 3.11 is the
  active interpreter (`python --version`). 3.12 / 3.13 have not been tested.

- **`docker: command not found`:** Docker Desktop (macOS / Windows) or Docker
  Engine (Linux) is not running or not on `PATH`. On macOS, open Docker
  Desktop and wait for "Docker is running" before retrying.

- **Port 5000 / 8000 / 9090 / 3000 already in use:** another process is using
  the port. Use a different port (e.g. `uvicorn src.api:app --port 8001`)
  or stop the conflicting process. macOS specifically reserves port 5000 for
  AirPlay Receiver; that is why MLflow uses 5050.

- **`docker build` fails with "models/best_model.pkl not found":** training
  has not been run yet. Run `make train` first, then retry the build.

- **Prometheus shows the `heart-disease-api` target as DOWN:** the scrape
  target is `host.docker.internal:8000`. The API must be running on the host
  on port 8000. Confirm with `curl http://localhost:8000/metrics` from the
  host.

- **`kubectl apply` fails with "connection refused":** there is no running
  cluster. Enable Kubernetes in Docker Desktop or start a `kind` cluster
  with `kind create cluster`.

- **Helm chart installed but pods are `ImagePullBackOff`:** the local image
  is not available to the cluster. With `kind`, run
  `kind load docker-image heart-disease-api:1.0.0` and re-deploy.

- **`make: command not found` (Windows):** install via Chocolatey
  (`choco install make`) or run the underlying commands directly — they
  are listed in each section above.
