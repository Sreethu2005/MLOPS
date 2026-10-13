# Makefile for Hospital Readmission MLOps Pipeline
PYTHON ?= python

.PHONY: help data features train explain registry monitor retrain test docker-build docker-up serve dashboard all

help:
	@echo "Hospital Readmission MLOps Pipeline Commands:"
	@echo "  make data         - Ingest, clean, anti-leakage split & validate data"
	@echo "  make features     - Fit & serialize feature preprocessor"
	@echo "  make train        - Train LR, RF, and XGBoost models with Platt calibration"
	@echo "  make explain      - Compute TreeSHAP summary & local attributions"
	@echo "  make registry     - Register champion model in MLflow Model Registry"
	@echo "  make monitor      - Run PSI drift detection & demographic fairness audit"
	@echo "  make retrain      - Execute automated retraining & quality gates"
	@echo "  make test         - Run full pytest test suite (35+ unit & integration tests)"
	@echo "  make serve        - Run FastAPI serving application locally on port 8000"
	@echo "  make dashboard    - Run Streamlit decision-support dashboard on port 8501"
	@echo "  make docker-build - Build hospital-readmission-api Docker image"
	@echo "  make docker-up    - Start full docker-compose stack (API + MLflow + Postgres)"
	@echo "  make all          - Run complete end-to-end pipeline and test suite"

data:
	$(PYTHON) run_pipeline.py --step data

features:
	$(PYTHON) run_pipeline.py --step features

train:
	$(PYTHON) run_pipeline.py --step train

explain:
	$(PYTHON) run_pipeline.py --step explain

registry:
	$(PYTHON) run_pipeline.py --step registry

monitor:
	$(PYTHON) run_pipeline.py --step monitor

retrain:
	$(PYTHON) run_pipeline.py --step retrain

test:
	$(PYTHON) -m pytest -v tests/

serve:
	$(PYTHON) -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

docker-build:
	docker build -t hospital-readmission-api:latest -f docker/Dockerfile.api .

docker-up:
	docker compose up -d api mlflow

all:
	$(PYTHON) run_pipeline.py --step all
