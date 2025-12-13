# =============================================================================
# AlphaTrade System - Makefile
# =============================================================================
#
# Easy command interface for the trading system.
#
# Usage:
#   make help          # Show all available commands
#   make pipeline      # Run full pipeline
#   make paper         # Start paper trading
#
# =============================================================================

.PHONY: help data features labels train calibrate backtest validate pipeline pipeline-fast paper live test clean docker-build docker-up docker-down docker-logs health setup

# Default Python - can override with PYTHON=python3.10 make ...
PYTHON ?= python

# =============================================================================
# HELP
# =============================================================================

help:
	@echo "============================================================================="
	@echo "AlphaTrade System - Available Commands"
	@echo "============================================================================="
	@echo ""
	@echo "Data & Features:"
	@echo "  make data          Download and preprocess historical data"
	@echo "  make features      Generate institutional features"
	@echo "  make labels        Create triple barrier labels"
	@echo ""
	@echo "Training:"
	@echo "  make train         Train CatBoost model with CV"
	@echo "  make calibrate     Calibrate probabilities for Kelly"
	@echo "  make train-all     labels + train + calibrate"
	@echo ""
	@echo "Backtesting:"
	@echo "  make backtest      Run backtest with realistic fills"
	@echo "  make validate      Validate all components"
	@echo ""
	@echo "Full Pipeline:"
	@echo "  make pipeline      Run entire pipeline (data -> validate)"
	@echo "  make pipeline-fast Skip data if exists, retrain model"
	@echo ""
	@echo "Trading:"
	@echo "  make paper         Start paper trading"
	@echo "  make live          Start live trading (requires confirmation)"
	@echo ""
	@echo "Utilities:"
	@echo "  make health        Check system health"
	@echo "  make test          Run unit tests"
	@echo "  make clean         Remove generated files"
	@echo "  make setup         Initial setup"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-up     Start all services"
	@echo "  make docker-down   Stop all services"
	@echo "  make docker-logs   View logs"
	@echo ""
	@echo "============================================================================="

# =============================================================================
# DATA & FEATURES
# =============================================================================

data:
	@echo "Downloading and preprocessing data..."
	$(PYTHON) scripts/run_pipeline.py --stage data

features:
	@echo "Generating institutional features..."
	$(PYTHON) scripts/run_pipeline.py --stage features

labels:
	@echo "Creating triple barrier labels..."
	$(PYTHON) scripts/run_pipeline.py --stage labels

# =============================================================================
# TRAINING
# =============================================================================

train:
	@echo "Training CatBoost model..."
	$(PYTHON) scripts/run_pipeline.py --stage train

calibrate:
	@echo "Calibrating probabilities..."
	$(PYTHON) scripts/run_pipeline.py --stage calibrate

train-all: labels train calibrate
	@echo "Training pipeline complete!"

# =============================================================================
# BACKTESTING & VALIDATION
# =============================================================================

backtest:
	@echo "Running backtest with realistic fills..."
	$(PYTHON) scripts/run_pipeline.py --stage backtest

validate:
	@echo "Validating all components..."
	$(PYTHON) scripts/run_pipeline.py --stage validate

# =============================================================================
# FULL PIPELINE
# =============================================================================

pipeline:
	@echo "Running full pipeline..."
	$(PYTHON) scripts/run_pipeline.py

pipeline-fast:
	@echo "Running fast pipeline (skip data if exists)..."
	$(PYTHON) scripts/run_pipeline.py --skip-data --force

# =============================================================================
# TRADING
# =============================================================================

paper:
	@echo "Starting paper trading..."
	@echo "Press Ctrl+C to stop"
	$(PYTHON) main.py --mode paper

live:
	@echo "=============================================="
	@echo "WARNING: LIVE TRADING MODE"
	@echo "This will trade with REAL money!"
	@echo "=============================================="
	@read -p "Type 'CONFIRM' to proceed: " confirm; \
	if [ "$$confirm" = "CONFIRM" ]; then \
		$(PYTHON) main.py --mode live; \
	else \
		echo "Aborted."; \
	fi

# =============================================================================
# UTILITIES
# =============================================================================

health:
	@echo "Checking system health..."
	$(PYTHON) -c "from main import AlphaTradeSystem; import asyncio; s = AlphaTradeSystem(); asyncio.run(s.initialize()); print(s.get_health_status())"

test:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	@echo "Running linter..."
	$(PYTHON) -m flake8 src/ scripts/ main.py --max-line-length=120 --ignore=E501,W503

format:
	@echo "Formatting code..."
	$(PYTHON) -m black src/ scripts/ main.py --line-length=120

setup:
	@echo "Running initial setup..."
	@if [ -f scripts/setup.sh ]; then \
		bash scripts/setup.sh; \
	else \
		echo "Creating directories..."; \
		mkdir -p data/raw data/cache data/processed data/holdout; \
		mkdir -p results/features results/labels results/backtest results/paper; \
		mkdir -p models logs; \
		echo "Installing dependencies..."; \
		$(PYTHON) -m pip install -r requirements.txt; \
		echo "Setup complete!"; \
	fi

clean:
	@echo "Cleaning generated files..."
	rm -rf results/
	rm -rf logs/*.log
	rm -rf __pycache__
	rm -rf src/**/__pycache__
	rm -rf .pytest_cache
	rm -f models/calibration_model.pkl
	@echo "Clean complete! (models/model.pkl preserved)"

clean-all: clean
	@echo "Removing all generated files including models..."
	rm -rf models/*.pkl
	rm -rf data/cache/
	rm -rf data/processed/
	@echo "Full clean complete!"

# =============================================================================
# DOCKER
# =============================================================================

docker-build:
	@echo "Building Docker image..."
	docker build -t alphatrade:latest .

docker-up:
	@echo "Starting Docker services..."
	docker-compose up -d
	@echo "Services started!"
	@echo "  - Grafana: http://localhost:3000"
	@echo "  - Prometheus: http://localhost:9090"

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-shell:
	docker-compose exec alphatrade /bin/bash

docker-restart:
	docker-compose restart alphatrade

# =============================================================================
# DEVELOPMENT
# =============================================================================

dev-install:
	@echo "Installing development dependencies..."
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install pytest pytest-asyncio black flake8 mypy

jupyter:
	@echo "Starting Jupyter notebook..."
	$(PYTHON) -m jupyter notebook --notebook-dir=notebooks

# =============================================================================
# QUICK COMMANDS
# =============================================================================

# Train on a single symbol for quick testing
train-single:
	$(PYTHON) scripts/train_models.py --symbol AAPL --n-estimators 50

# Quick backtest on recent data
backtest-quick:
	$(PYTHON) main.py --mode backtest

# Check if model is trained
check-model:
	@if [ -f models/model.pkl ]; then \
		echo "Model exists: models/model.pkl"; \
		cat models/metrics.yaml | head -20; \
	else \
		echo "No model found. Run 'make train' first."; \
	fi

# Show current configuration
show-config:
	@echo "=== Settings ===" && cat config/settings.yaml 2>/dev/null || echo "Not found"
	@echo ""
	@echo "=== Risk Params ===" && cat config/risk_params.yaml 2>/dev/null || echo "Not found"

# =============================================================================
# CI/CD
# =============================================================================

ci-test: lint test

ci-build: docker-build

ci-deploy: ci-test ci-build
	@echo "Ready for deployment!"
