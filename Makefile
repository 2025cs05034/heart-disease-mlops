SHELL := /bin/bash
PYTHON ?= python
IMAGE  ?= heart-disease-api:local
PORT   ?= 8000
MLFLOW_PORT ?= 5050

.PHONY: help install lint format test data train mlflow-ui api-run \
        docker-build docker-run docker-stop \
        monitoring-up monitoring-down monitoring-logs \
        k8s-apply k8s-delete helm-install helm-uninstall \
        clean clean-data clean-mlflow

help:
	@echo "Heart Disease MLOps - common make targets"
	@echo ""
	@echo "  Setup"
	@echo "    install            Install Python dependencies"
	@echo ""
	@echo "  Code quality"
	@echo "    lint               Run flake8, black --check, isort --check"
	@echo "    format             Auto-format with black + isort"
	@echo "    test               Run pytest with coverage"
	@echo ""
	@echo "  Data + training"
	@echo "    data               Download UCI Heart Disease dataset"
	@echo "    train              Train models and log to MLflow"
	@echo "    mlflow-ui          Launch MLflow UI on :$(MLFLOW_PORT)"
	@echo ""
	@echo "  API"
	@echo "    api-run            Run FastAPI locally (uvicorn) on :$(PORT)"
	@echo "    docker-build       Build the API Docker image"
	@echo "    docker-run         Run the API container on :$(PORT)"
	@echo "    docker-stop        Stop and remove the API container"
	@echo ""
	@echo "  Monitoring"
	@echo "    monitoring-up      Start Prometheus + Grafana via docker compose"
	@echo "    monitoring-down    Stop the monitoring stack"
	@echo "    monitoring-logs    Tail logs from the monitoring stack"
	@echo ""
	@echo "  Deployment"
	@echo "    k8s-apply          Apply raw Kubernetes manifests"
	@echo "    k8s-delete         Delete the K8s deployment"
	@echo "    helm-install       Install the Helm chart"
	@echo "    helm-uninstall     Uninstall the Helm release"
	@echo ""
	@echo "  Cleanup"
	@echo "    clean              Remove caches and coverage artifacts"
	@echo "    clean-data         Remove downloaded raw data"
	@echo "    clean-mlflow       Remove local MLflow store"

install:
	@echo "==> Installing Python dependencies"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	@echo "==> flake8"
	flake8 src tests
	@echo "==> black --check"
	black --check src tests
	@echo "==> isort --check"
	isort --check-only src tests

format:
	@echo "==> Formatting with black + isort"
	black src tests
	isort src tests

test:
	@echo "==> Running pytest with coverage"
	pytest --cov=src --cov-report=term-missing tests/

data:
	@echo "==> Downloading UCI Heart Disease dataset"
	$(PYTHON) -m src.download_data

train: data
	@echo "==> Training models (logging to MLflow at sqlite:///mlflow.db)"
	$(PYTHON) -m src.train

mlflow-ui:
	@echo "==> MLflow UI on http://localhost:$(MLFLOW_PORT)"
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

api-run:
	@echo "==> uvicorn on http://localhost:$(PORT)"
	uvicorn src.api:app --host 0.0.0.0 --port $(PORT) --reload

docker-build:
	@echo "==> Building $(IMAGE)"
	docker build -t $(IMAGE) .

docker-run:
	@echo "==> Running $(IMAGE) on http://localhost:$(PORT)"
	docker run -d --name heart-disease-api -p $(PORT):8000 $(IMAGE)
	@echo "    Try: curl http://localhost:$(PORT)/health"

docker-stop:
	@echo "==> Stopping API container"
	-docker stop heart-disease-api
	-docker rm heart-disease-api

monitoring-up:
	@echo "==> Starting Prometheus + Grafana stack"
	docker compose -f deploy/monitoring/docker-compose.monitoring.yml up -d
	@echo "    Prometheus: http://localhost:9090"
	@echo "    Grafana:    http://localhost:3000"

monitoring-down:
	@echo "==> Stopping monitoring stack"
	docker compose -f deploy/monitoring/docker-compose.monitoring.yml down

monitoring-logs:
	docker compose -f deploy/monitoring/docker-compose.monitoring.yml logs -f

k8s-apply:
	@echo "==> Applying Kubernetes manifests"
	kubectl apply -f deploy/k8s/namespace.yaml
	kubectl apply -f deploy/k8s/

k8s-delete:
	@echo "==> Deleting Kubernetes resources"
	kubectl delete -f deploy/k8s/ --ignore-not-found

helm-install:
	@echo "==> Installing Helm chart"
	helm upgrade --install heart-disease deploy/helm/heart-disease-api \
		--namespace heart-disease --create-namespace

helm-uninstall:
	@echo "==> Uninstalling Helm release"
	helm uninstall heart-disease --namespace heart-disease

clean:
	@echo "==> Cleaning caches and coverage"
	rm -rf .pytest_cache .coverage htmlcov coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-data:
	@echo "==> Removing downloaded raw data"
	rm -rf data/raw/*.csv

clean-mlflow:
	@echo "==> Removing local MLflow store"
	rm -rf mlruns mlartifacts mlflow.db
