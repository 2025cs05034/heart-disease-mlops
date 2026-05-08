# heart-disease-mlops

MLOps assignment (BITS Pilani, M.Tech, S2-25_AMLCSZG523).

Trains a heart-disease classifier on the UCI Cleveland dataset (303 records),
serves it through FastAPI inside a Docker container, and runs it on Kubernetes
with Prometheus + Grafana for monitoring.

Best model so far: Logistic Regression, test ROC-AUC `0.9588`,
test accuracy `0.8688`.

## Setup

Tested on macOS (Apple Silicon) with Python 3.11 and Docker Desktop.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Most common tasks are wrapped in the Makefile. Run `make help` to see them.
The rest of this README spells the commands out so they work without `make`.

## Train

```bash
python -m src.download_data   # writes data/raw/heart_disease.csv
python -m src.train            # trains LR + RF, picks the best
```

Training logs to MLflow with a SQLite backend. The best pipeline (preprocessor
+ estimator) is saved to `models/best_model.pkl`. To browse runs:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5050
# macOS port 5000 is taken by AirPlay, that's why 5050.
```

## API

```bash
uvicorn src.api:app --port 8000
```

Endpoints:

- `GET /health` and `GET /ready` for probes
- `POST /predict` for a single prediction
- `POST /predict/batch` for many at once
- `GET /metrics` for Prometheus

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

## Tests and lint

```bash
pytest tests/
flake8 src tests
black --check src tests
isort --check-only src tests
```

CI runs the same four steps plus a Docker build smoke test on every push, see
`.github/workflows/ci.yml`.

## Docker

```bash
docker build -t heart-disease-api:1.0.0 .
docker run --rm -p 8000:8000 heart-disease-api:1.0.0
```

The image runs as UID 1001 (non-root) and ships with a Python healthcheck.

## Kubernetes

Two flavors. Use whichever you prefer.

Raw manifests:

```bash
kubectl apply -f deploy/k8s/
kubectl get all -n heart-disease
```

Helm:

```bash
kubectl create namespace heart-disease
helm install heart-disease-api deploy/helm/heart-disease-api \
     --namespace heart-disease
```

Both options ship a Deployment, Service (LoadBalancer), Ingress, an HPA
(2-5 replicas, 70% CPU target) and a default-deny NetworkPolicy.

## Monitoring

Local stack via Docker Compose:

```bash
cp deploy/monitoring/grafana_admin_password.txt.example \
   deploy/monitoring/grafana_admin_password.txt
# edit the file, set your own admin password, then:
docker compose -f deploy/monitoring/docker-compose.monitoring.yml up -d
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

A Grafana dashboard called "Heart Disease API" is auto-provisioned from
`deploy/monitoring/grafana/dashboards/`. It plots request rate, latency
percentiles, total predictions and disease vs no-disease counts.

## Repo layout

```
src/                 training and inference code
tests/               pytest suite (23 tests)
notebooks/           EDA notebook (executed)
deploy/k8s/          Kubernetes manifests
deploy/helm/         Helm chart
deploy/monitoring/   Prometheus + Grafana stack
.github/workflows/   CI pipeline
screenshots/         screenshots used in the report
```

## Dataset

UCI Heart Disease, Cleveland subset.
https://archive.ics.uci.edu/dataset/45/heart+disease

`src/download_data.py` fetches the raw CSV, falls back to the `ucimlrepo`
package if the direct URL fails, drops malformed rows, coerces feature
columns to numeric, and collapses the 0-4 severity target into a binary
0/1 label.
