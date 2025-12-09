# AlphaTrade Platform

**JPMorgan-Level Algorithmic Trading System**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

AlphaTrade is a production-grade, distributed algorithmic trading platform designed with institutional-level architecture. It supports backtesting, paper trading, and live trading with ML-powered strategies.

### Key Features

- **Distributed Microservices Architecture** - Event-driven services communicating via Redis pub/sub
- **ML-Powered Strategies** - LightGBM, XGBoost, and ensemble models with real-time inference
- **Smart Order Routing** - TWAP, VWAP, Iceberg, and Pegged execution algorithms
- **Risk Management** - Real-time position monitoring with automatic kill switch
- **MLOps Integration** - MLflow for model versioning and experiment tracking
- **High-Performance Data Pipeline** - TimescaleDB for time-series data, Redis for state

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     API Gateway (FastAPI)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │Data Ingestion│  │Strategy      │  │Risk Engine   │          │
│  │Service       │──│Engine        │──│              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Redis Message Bus (Pub/Sub)                │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                │                │                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │OEMS          │  │Feature Store │  │Watchdog      │          │
│  │              │──│              │──│Kill Switch   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │Redis State   │  │TimescaleDB   │  │MLflow        │          │
│  │Store         │  │              │  │Registry      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Redis 7.0+
- PostgreSQL 15+ with TimescaleDB
- Docker & Docker Compose (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AlphaTrade_System.git
cd AlphaTrade_System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your API keys and settings
```

### Running with Docker

```bash
cd docker
docker compose up -d
```

### Running in Development Mode

```bash
# Start all services in a single process
python main_distributed.py dev

# Or start specific services
python main_distributed.py start --service data_ingestion
python main_distributed.py start --service strategy_engine
python main_distributed.py start --service risk_engine
python main_distributed.py start --service oems
```

### Running Backtests

```bash
# Interactive mode
python main.py

# Command line
python main.py backtest --symbol AAPL MSFT GOOGL --strategy alpha_ml
```

### Check System Status

```bash
python main_distributed.py status
```

## Project Structure

```
AlphaTrade_System/
├── api/                    # FastAPI REST API
├── backtesting/            # Backtesting engine
├── config/                 # Configuration management
├── core/                   # Core types and interfaces
├── data/                   # Data loading and processing
├── docker/                 # Docker configuration
├── execution/              # Order execution algorithms
├── features/               # Feature engineering
├── infrastructure/         # Message bus, state store, pools
├── models/                 # ML models and training
├── portfolio/              # Portfolio optimization
├── risk/                   # Risk management
├── scripts/                # Utility scripts
├── services/               # Microservices
├── strategies/             # Trading strategies
├── tests/                  # Unit and integration tests
├── main.py                 # Monolithic entry point
├── main_distributed.py     # Distributed system entry point
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
└── .gitignore             # Git ignore rules
```

## Services

| Service | Description | Port |
|---------|-------------|------|
| Data Ingestion | Market data streaming via Alpaca WebSocket | - |
| Strategy Engine | ML model inference and signal generation | - |
| Risk Engine | Position validation and risk limits | - |
| OEMS | Order Execution Management System | - |
| Watchdog | Health monitoring and kill switch | - |
| API Gateway | REST API and WebSocket | 8000 |

## Configuration

All configuration is done via environment variables. See `.env.example` for available options:

- **Trading**: Mode, broker API keys, risk limits
- **Infrastructure**: Redis, TimescaleDB, MLflow URLs
- **Execution**: Default algorithms, TWAP/Iceberg settings
- **Monitoring**: Prometheus metrics, logging

## Trading Modes

1. **Backtest** - Historical simulation with realistic slippage/commission
2. **Paper** - Live paper trading with Alpaca (no real money)
3. **Live** - Production trading with real capital (use with caution!)

## Risk Management

The platform includes multiple layers of risk protection:

- **Position Limits** - Maximum size per symbol and total exposure
- **Loss Limits** - Daily loss and drawdown thresholds
- **Kill Switch** - Automatic emergency shutdown
- **Watchdog Service** - Independent health monitoring

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | System status |
| `/services` | GET | List registered services |
| `/positions` | GET | Current positions |
| `/risk` | GET | Risk state |
| `/killswitch` | POST | Trigger emergency shutdown |
| `/backtest` | POST | Run backtest |
| `/models` | GET | List ML models |

## Development

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific module
pytest tests/test_strategies.py -v
```

### Code Quality

```bash
# Lint
ruff check .

# Format
black .
isort .

# Type check
mypy .
```

## Supported Symbols

The platform supports 46+ symbols including:

- **Tech**: AAPL, MSFT, GOOGL, AMZN, META, NVDA, etc.
- **Finance**: JPM, BAC, GS, MS, etc.
- **Healthcare**: JNJ, UNH, PFE, etc.
- **ETFs**: SPY, QQQ, IWM, etc.
- **Crypto**: BTC/USD, ETH/USD (via Alpaca)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

**This software is for educational purposes only. Use at your own risk.**

Trading financial instruments carries significant risk of loss. Past performance is not indicative of future results. Always paper trade first and never risk money you cannot afford to lose.

## Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/yourusername/AlphaTrade_System/issues)
