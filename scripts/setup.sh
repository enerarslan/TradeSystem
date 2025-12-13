#!/bin/bash
# =============================================================================
# AlphaTrade System - One-Command Setup Script
# =============================================================================
#
# Usage:
#   ./scripts/setup.sh              # Full setup
#   ./scripts/setup.sh --no-docker  # Skip Docker setup
#   ./scripts/setup.sh --dev        # Include development dependencies
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_DOCKER=false
DEV_MODE=false

for arg in "$@"; do
    case $arg in
        --no-docker)
            SKIP_DOCKER=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
    esac
done

echo ""
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}       AlphaTrade System - Setup${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# =============================================================================
# STEP 1: Check Prerequisites
# =============================================================================

echo -e "${YELLOW}[1/8] Checking prerequisites...${NC}"

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}ERROR: Python not found!${NC}"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}ERROR: Python 3.10+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

echo -e "  ${GREEN}✓${NC} Python $PYTHON_VERSION"

# Check pip
if ! $PYTHON -m pip --version &> /dev/null; then
    echo -e "${RED}ERROR: pip not found!${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} pip installed"

# Check Docker (optional)
if [ "$SKIP_DOCKER" = false ]; then
    if command -v docker &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} Docker installed"
        DOCKER_AVAILABLE=true
    else
        echo -e "  ${YELLOW}⚠${NC} Docker not found (optional)"
        DOCKER_AVAILABLE=false
    fi

    if command -v docker-compose &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} Docker Compose installed"
    elif docker compose version &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} Docker Compose (plugin) installed"
    else
        echo -e "  ${YELLOW}⚠${NC} Docker Compose not found (optional)"
    fi
fi

# =============================================================================
# STEP 2: Create Virtual Environment
# =============================================================================

echo ""
echo -e "${YELLOW}[2/8] Setting up virtual environment...${NC}"

if [ -d "venv" ]; then
    echo -e "  ${GREEN}✓${NC} Virtual environment already exists"
else
    $PYTHON -m venv venv
    echo -e "  ${GREEN}✓${NC} Virtual environment created"
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

echo -e "  ${GREEN}✓${NC} Virtual environment activated"

# =============================================================================
# STEP 3: Install Dependencies
# =============================================================================

echo ""
echo -e "${YELLOW}[3/8] Installing dependencies...${NC}"

# Upgrade pip
pip install --upgrade pip -q

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    echo -e "  ${GREEN}✓${NC} Main dependencies installed"
else
    echo -e "${RED}ERROR: requirements.txt not found!${NC}"
    exit 1
fi

# Install development dependencies if requested
if [ "$DEV_MODE" = true ]; then
    pip install pytest pytest-asyncio black flake8 mypy jupyter -q
    echo -e "  ${GREEN}✓${NC} Development dependencies installed"
fi

# =============================================================================
# STEP 4: Create Directory Structure
# =============================================================================

echo ""
echo -e "${YELLOW}[4/8] Creating directory structure...${NC}"

mkdir -p data/raw
mkdir -p data/cache
mkdir -p data/processed
mkdir -p data/holdout
mkdir -p results/features
mkdir -p results/labels
mkdir -p results/backtest
mkdir -p results/paper
mkdir -p models
mkdir -p logs
mkdir -p notebooks

echo -e "  ${GREEN}✓${NC} Directories created"

# =============================================================================
# STEP 5: Setup Configuration Files
# =============================================================================

echo ""
echo -e "${YELLOW}[5/8] Setting up configuration files...${NC}"

# Copy example configs if they don't exist
if [ -f "config/settings.example.yaml" ] && [ ! -f "config/settings.yaml" ]; then
    cp config/settings.example.yaml config/settings.yaml
    echo -e "  ${GREEN}✓${NC} Created config/settings.yaml"
fi

if [ -f "config/symbols.example.yaml" ] && [ ! -f "config/symbols.yaml" ]; then
    cp config/symbols.example.yaml config/symbols.yaml
    echo -e "  ${GREEN}✓${NC} Created config/symbols.yaml"
fi

if [ -f ".env.example" ] && [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "  ${GREEN}✓${NC} Created .env"
fi

echo -e "  ${GREEN}✓${NC} Configuration files ready"

# =============================================================================
# STEP 6: Configure API Keys
# =============================================================================

echo ""
echo -e "${YELLOW}[6/8] Configuring API keys...${NC}"

# Check if API keys are already set
if [ -z "$ALPACA_API_KEY" ]; then
    echo ""
    echo "Alpaca API credentials are required for paper/live trading."
    echo "You can get free API keys at: https://app.alpaca.markets/signup"
    echo ""

    read -p "Enter Alpaca API Key (or press Enter to skip): " ALPACA_KEY

    if [ ! -z "$ALPACA_KEY" ]; then
        read -p "Enter Alpaca API Secret: " ALPACA_SECRET

        # Save to .env file
        if [ -f ".env" ]; then
            # Update existing .env
            sed -i.bak "s/^ALPACA_API_KEY=.*/ALPACA_API_KEY=$ALPACA_KEY/" .env
            sed -i.bak "s/^ALPACA_API_SECRET=.*/ALPACA_API_SECRET=$ALPACA_SECRET/" .env
        else
            # Create new .env
            echo "ALPACA_API_KEY=$ALPACA_KEY" > .env
            echo "ALPACA_API_SECRET=$ALPACA_SECRET" >> .env
        fi

        echo -e "  ${GREEN}✓${NC} API keys saved to .env"
    else
        echo -e "  ${YELLOW}⚠${NC} API keys skipped (required for trading)"
    fi
else
    echo -e "  ${GREEN}✓${NC} API keys already configured"
fi

# =============================================================================
# STEP 7: Start Docker Services (Optional)
# =============================================================================

echo ""
echo -e "${YELLOW}[7/8] Setting up Docker services...${NC}"

if [ "$SKIP_DOCKER" = true ]; then
    echo -e "  ${YELLOW}⚠${NC} Docker setup skipped (--no-docker)"
elif [ "$DOCKER_AVAILABLE" = true ]; then
    read -p "Start Docker services (Redis, Grafana, Prometheus)? [y/N]: " start_docker

    if [ "$start_docker" = "y" ] || [ "$start_docker" = "Y" ]; then
        if [ -f "docker-compose.yaml" ]; then
            docker-compose up -d redis grafana prometheus 2>/dev/null || docker compose up -d redis grafana prometheus
            echo -e "  ${GREEN}✓${NC} Docker services started"
            echo -e "       Grafana: http://localhost:3000"
            echo -e "       Prometheus: http://localhost:9090"
        else
            echo -e "  ${YELLOW}⚠${NC} docker-compose.yaml not found"
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} Docker services not started"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Docker not available"
fi

# =============================================================================
# STEP 8: Validate Setup
# =============================================================================

echo ""
echo -e "${YELLOW}[8/8] Validating setup...${NC}"

# Test imports
$PYTHON -c "
import sys
sys.path.insert(0, '.')
try:
    from src.utils.logger import get_logger
    from src.data.loader import MultiAssetLoader
    from src.features.institutional import InstitutionalFeatureEngineer
    print('  ✓ Core imports successful')
except Exception as e:
    print(f'  ✗ Import error: {e}')
    sys.exit(1)

try:
    from main import AlphaTradeSystem
    print('  ✓ Main module imports successful')
except Exception as e:
    print(f'  ✗ Main import error: {e}')
    sys.exit(1)
" 2>/dev/null || {
    echo -e "  ${RED}✗${NC} Import validation failed"
    echo "  Check the error messages above"
}

# Check for existing model
if [ -f "models/model.pkl" ]; then
    echo -e "  ${GREEN}✓${NC} Trained model found"
else
    echo -e "  ${YELLOW}⚠${NC} No trained model (run 'make pipeline' to train)"
fi

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the virtual environment:"
echo -e "     ${YELLOW}source venv/bin/activate${NC}  (Linux/Mac)"
echo -e "     ${YELLOW}venv\\Scripts\\activate${NC}    (Windows)"
echo ""
echo "  2. Run the full pipeline (trains model, runs backtest):"
echo -e "     ${YELLOW}make pipeline${NC}"
echo ""
echo "  3. Or skip to paper trading (if model exists):"
echo -e "     ${YELLOW}make paper${NC}"
echo ""
echo "  4. View all available commands:"
echo -e "     ${YELLOW}make help${NC}"
echo ""
echo -e "${BLUE}=============================================================================${NC}"
echo ""
