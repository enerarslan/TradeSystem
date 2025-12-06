# =============================================================================
# ALGO TRADING PLATFORM - MAKEFILE
# =============================================================================
# Usage: make <target>
# =============================================================================

.PHONY: help install install-dev test lint format clean backtest paper api docs

# Default target
help:
	@echo "Algo Trading Platform - Available Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make setup        Full setup (install + create dirs)"
	@echo ""
	@echo "Development:"
	@echo "  make test         Run all tests"
	@echo "  make test-fast    Run tests without slow tests"
	@echo "  make lint         Run linters (ruff, mypy)"
	@echo "  make format       Format code (black, isort)"
	@echo "  make check        Run all checks (lint + test)"
	@echo ""
	@echo "Trading:"
	@echo "  make backtest     Run backtest (interactive)"
	@echo "  make paper        Start paper trading"
	@echo "  make api          Start API server"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        Remove cache and build files"
	@echo "  make docs         Generate documentation"
	@echo ""

# =============================================================================
# SETUP
# =============================================================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt 2>/dev/null || true
	pip install ruff black isort mypy pre-commit

setup: install
	mkdir -p data/storage data/processed data/cache
	mkdir -p models/artifacts
	mkdir -p backtesting/reports
	mkdir -p logs
	cp .env.example .env 2>/dev/null || true
	@echo "Setup complete! Edit .env with your configuration."

# =============================================================================
# DEVELOPMENT
# =============================================================================

test:
	python -m pytest tests/ -v --tb=short

test-fast:
	python -m pytest tests/ -v --tb=short -m "not slow"

test-cov:
	python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term

lint:
	ruff check .
	mypy . --ignore-missing-imports 2>/dev/null || true

format:
	black .
	isort .
	ruff check --fix .

check: lint test

# =============================================================================
# TRADING
# =============================================================================

backtest:
	python main.py backtest

backtest-all:
	python main.py backtest --all-symbols

paper:
	python main.py paper

api:
	python main.py api

# =============================================================================
# UTILITIES
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage 2>/dev/null || true

docs:
	@echo "Documentation generation not yet implemented"

# =============================================================================
# DOCKER (Future)
# =============================================================================

docker-build:
	docker build -t algo-trading-platform .

docker-run:
	docker run -it --rm -v $(PWD)/data:/app/data algo-trading-platform

# =============================================================================
# DATA
# =============================================================================

download-sample-data:
	@echo "Downloading sample data..."
	@echo "Note: Add your data files to data/storage/"

validate-data:
	python -c "from data.loader import CSVLoader; loader = CSVLoader(); print(loader.list_files())"
