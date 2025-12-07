"""
FastAPI Application
===================

REST API for the algorithmic trading platform.

Provides endpoints for:
- Backtesting
- Model training and inference
- Strategy management
- Data access
- Portfolio monitoring
- Real-time WebSocket updates

Author: Algo Trading Platform
License: MIT
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

import numpy as np
import polars as pl


# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)


class BacktestRequest(BaseModel):
    """Backtest request parameters."""
    symbols: list[str]
    strategy: str = "trend_following"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005


class BacktestResponse(BaseModel):
    """Backtest result response."""
    task_id: str
    status: str = "queued"
    message: str = "Backtest queued for execution"


class BacktestResult(BaseModel):
    """Backtest result."""
    task_id: str
    status: str
    total_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_trades: Optional[int] = None
    win_rate: Optional[float] = None
    error: Optional[str] = None


class TrainRequest(BaseModel):
    """Model training request."""
    symbols: list[str]
    model_type: str = "lightgbm"
    optimize: bool = True
    n_trials: int = 50
    target_horizon: int = 5


class TrainResponse(BaseModel):
    """Training result response."""
    task_id: str
    status: str = "queued"
    message: str = "Training queued for execution"


class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    name: str
    model_type: str
    created_at: datetime
    accuracy: Optional[float] = None
    feature_count: int = 0


class PredictRequest(BaseModel):
    """Prediction request."""
    model_id: str
    symbol: str
    features: Optional[dict[str, float]] = None


class PredictResponse(BaseModel):
    """Prediction response."""
    symbol: str
    prediction: int
    confidence: float
    probabilities: dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.now)


class StrategyInfo(BaseModel):
    """Strategy information."""
    name: str
    description: str
    parameters: dict[str, Any]
    type: str


class PortfolioState(BaseModel):
    """Portfolio state."""
    equity: float
    cash: float
    positions: dict[str, dict[str, Any]]
    unrealized_pnl: float
    total_return: float
    timestamp: datetime = Field(default_factory=datetime.now)


class DataRequest(BaseModel):
    """Data request."""
    symbol: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timeframe: str = "15min"


# =============================================================================
# APPLICATION STATE
# =============================================================================

class AppState:
    """Application state manager."""
    
    def __init__(self):
        self.tasks: dict[str, dict[str, Any]] = {}
        self.models: dict[str, Any] = {}
        self.websockets: list[WebSocket] = []
        self.is_initialized: bool = False
    
    async def initialize(self):
        """Initialize application resources."""
        if not self.is_initialized:
            # Create necessary directories
            Path("models/artifacts").mkdir(parents=True, exist_ok=True)
            Path("data/storage").mkdir(parents=True, exist_ok=True)
            Path("backtesting/reports").mkdir(parents=True, exist_ok=True)
            
            self.is_initialized = True
    
    async def shutdown(self):
        """Cleanup resources."""
        # Close all websocket connections
        for ws in self.websockets:
            try:
                await ws.close()
            except Exception:
                pass
        self.websockets.clear()


state = AppState()


# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    await state.initialize()
    yield
    # Shutdown
    await state.shutdown()


# =============================================================================
# APPLICATION FACTORY
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Algo Trading Platform API",
        description="JPMorgan-level algorithmic trading platform",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {"message": "Algo Trading Platform API", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@app.get("/status", tags=["Health"])
async def system_status():
    """Get system status."""
    return {
        "status": "operational",
        "initialized": state.is_initialized,
        "active_tasks": len([t for t in state.tasks.values() if t["status"] == "running"]),
        "loaded_models": len(state.models),
        "websocket_connections": len(state.websockets),
    }


# =============================================================================
# BACKTEST ENDPOINTS
# =============================================================================

@app.post("/backtest", response_model=BacktestResponse, tags=["Backtest"])
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Queue a backtest for execution."""
    task_id = str(uuid4())
    
    state.tasks[task_id] = {
        "type": "backtest",
        "status": "queued",
        "request": request.model_dump(),
        "created_at": datetime.now().isoformat(),
        "result": None,
    }
    
    # Add to background tasks
    background_tasks.add_task(execute_backtest, task_id, request)
    
    return BacktestResponse(task_id=task_id)


async def execute_backtest(task_id: str, request: BacktestRequest):
    """Execute backtest in background."""
    try:
        state.tasks[task_id]["status"] = "running"
        
        # Import here to avoid circular imports
        from config.settings import get_settings
        from data.loader import CSVLoader
        from data.processor import DataProcessor
        from backtesting.engine import BacktestEngine, BacktestConfig
        from strategies import create_strategy
        
        settings = get_settings()
        loader = CSVLoader(storage_path=settings.data.storage_path)
        processor = DataProcessor()
        
        # Load data
        data = {}
        for symbol in request.symbols:
            try:
                df = loader.load(symbol)
                df = processor.process(df)
                data[symbol] = df
            except Exception as e:
                pass
        
        if not data:
            raise ValueError("No data could be loaded")
        
        # Create strategy
        strategy = create_strategy(request.strategy, list(data.keys()))
        
        # Configure backtest
        config = BacktestConfig(
            initial_capital=request.initial_capital,
            commission_pct=request.commission_pct,
            slippage_pct=request.slippage_pct,
        )
        
        # Run backtest
        engine = BacktestEngine(config)
        for symbol, df in data.items():
            engine.add_data(symbol, df)
        engine.add_strategy(strategy)
        
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d") if request.start_date else None
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d") if request.end_date else None
        
        report = engine.run(start_date, end_date, show_progress=False)
        
        # Store result
        state.tasks[task_id]["status"] = "completed"
        state.tasks[task_id]["result"] = {
            "total_return": report.total_return_pct,
            "sharpe_ratio": report.sharpe_ratio,
            "max_drawdown": report.max_drawdown,
            "total_trades": report.total_trades,
            "win_rate": report.win_rate,
        }
        
        # Notify websocket clients
        await broadcast_task_update(task_id)
        
    except Exception as e:
        state.tasks[task_id]["status"] = "failed"
        state.tasks[task_id]["error"] = str(e)


@app.get("/backtest/{task_id}", response_model=BacktestResult, tags=["Backtest"])
async def get_backtest_result(task_id: str):
    """Get backtest result by task ID."""
    if task_id not in state.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = state.tasks[task_id]
    result = task.get("result", {})
    
    return BacktestResult(
        task_id=task_id,
        status=task["status"],
        total_return=result.get("total_return"),
        sharpe_ratio=result.get("sharpe_ratio"),
        max_drawdown=result.get("max_drawdown"),
        total_trades=result.get("total_trades"),
        win_rate=result.get("win_rate"),
        error=task.get("error"),
    )


@app.get("/backtest", tags=["Backtest"])
async def list_backtests():
    """List all backtest tasks."""
    backtests = [
        {
            "task_id": task_id,
            "status": task["status"],
            "created_at": task["created_at"],
        }
        for task_id, task in state.tasks.items()
        if task["type"] == "backtest"
    ]
    return {"backtests": backtests}


# =============================================================================
# MODEL ENDPOINTS
# =============================================================================

@app.post("/models/train", response_model=TrainResponse, tags=["Models"])
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Queue model training."""
    task_id = str(uuid4())
    
    state.tasks[task_id] = {
        "type": "training",
        "status": "queued",
        "request": request.model_dump(),
        "created_at": datetime.now().isoformat(),
        "result": None,
    }
    
    background_tasks.add_task(execute_training, task_id, request)
    
    return TrainResponse(task_id=task_id)


async def execute_training(task_id: str, request: TrainRequest):
    """Execute model training in background."""
    try:
        state.tasks[task_id]["status"] = "running"
        
        from config.settings import get_settings
        from data.loader import CSVLoader
        from data.processor import DataProcessor
        from features.pipeline import FeaturePipeline, create_default_config
        from models.training import TrainingPipeline, TrainingConfig, OptimizationConfig
        
        settings = get_settings()
        loader = CSVLoader(storage_path=settings.data.storage_path)
        processor = DataProcessor()
        
        # Load and process data
        all_data = []
        for symbol in request.symbols:
            try:
                df = loader.load(symbol)
                df = processor.process(df)
                all_data.append(df)
            except Exception:
                pass
        
        if not all_data:
            raise ValueError("No data could be loaded")
        
        # Combine data
        combined_data = pl.concat(all_data)
        
        # Generate features
        pipeline = FeaturePipeline(create_default_config())
        df_features = pipeline.generate(combined_data)
        df_features = pipeline.create_target(df_features, horizon=request.target_horizon)
        X_train, X_test, y_train, y_test, feature_names = pipeline.prepare_train_test(df_features)
        
        # Train model
        opt_config = OptimizationConfig(n_trials=request.n_trials if request.optimize else 0)
        train_config = TrainingConfig(
            auto_optimize=request.optimize,
            optimization_config=opt_config,
        )
        
        training_pipeline = TrainingPipeline(train_config)
        model = training_pipeline.train(
            request.model_type,
            X_train, y_train,
            X_test, y_test,
            feature_names=feature_names,
        )
        
        # Save model
        model_id = str(uuid4())[:8]
        model_path = Path("models/artifacts") / f"api_{model_id}.pkl"
        model.save(model_path)
        
        # Store in state
        state.models[model_id] = {
            "path": str(model_path),
            "type": request.model_type,
            "created_at": datetime.now().isoformat(),
            "symbols": request.symbols,
        }
        
        # Update task
        metrics = model.evaluate(X_test, y_test)
        state.tasks[task_id]["status"] = "completed"
        state.tasks[task_id]["result"] = {
            "model_id": model_id,
            "accuracy": metrics.get("accuracy", 0),
            "feature_count": len(feature_names),
        }
        
    except Exception as e:
        state.tasks[task_id]["status"] = "failed"
        state.tasks[task_id]["error"] = str(e)


@app.get("/models", response_model=list[ModelInfo], tags=["Models"])
async def list_models():
    """List all available models."""
    models = []
    for model_id, info in state.models.items():
        models.append(ModelInfo(
            model_id=model_id,
            name=f"Model {model_id}",
            model_type=info["type"],
            created_at=datetime.fromisoformat(info["created_at"]),
        ))
    return models


@app.get("/models/{model_id}", tags=["Models"])
async def get_model_info(model_id: str):
    """Get model information."""
    if model_id not in state.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return state.models[model_id]


@app.post("/models/{model_id}/predict", response_model=PredictResponse, tags=["Models"])
async def predict(model_id: str, request: PredictRequest):
    """Make prediction with a model."""
    if model_id not in state.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        from models.base import BaseModel
        
        model_info = state.models[model_id]
        model = BaseModel.load(model_info["path"])
        
        # Get latest data for symbol
        from config.settings import get_settings
        from data.loader import CSVLoader
        from features.pipeline import FeaturePipeline, create_default_config
        
        settings = get_settings()
        loader = CSVLoader(storage_path=settings.data.storage_path)
        
        df = loader.load(request.symbol)
        
        pipeline = FeaturePipeline(create_default_config())
        df_features = pipeline.generate(df)
        
        # Get last row for prediction
        X = df_features.select(pipeline._feature_names).to_numpy()[-1:]
        
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else {}
        
        return PredictResponse(
            symbol=request.symbol,
            prediction=int(prediction),
            confidence=float(max(proba)) if proba else 0.5,
            probabilities={str(i): float(p) for i, p in enumerate(proba)} if len(proba) > 0 else {},
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_id}", tags=["Models"])
async def delete_model(model_id: str):
    """Delete a model."""
    if model_id not in state.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = state.models[model_id]
    Path(model_info["path"]).unlink(missing_ok=True)
    del state.models[model_id]
    
    return {"message": f"Model {model_id} deleted"}


# =============================================================================
# STRATEGY ENDPOINTS
# =============================================================================

@app.get("/strategies", response_model=list[StrategyInfo], tags=["Strategies"])
async def list_strategies():
    """List available strategies."""
    strategies = [
        StrategyInfo(
            name="trend_following",
            description="Trend following strategy using moving averages",
            parameters={"fast_period": 10, "slow_period": 30},
            type="momentum",
        ),
        StrategyInfo(
            name="mean_reversion",
            description="Mean reversion strategy using Bollinger Bands",
            parameters={"period": 20, "std_dev": 2.0},
            type="momentum",
        ),
        StrategyInfo(
            name="breakout",
            description="Breakout strategy using price channels",
            parameters={"period": 20},
            type="momentum",
        ),
        StrategyInfo(
            name="macd",
            description="MACD crossover strategy",
            parameters={"fast": 12, "slow": 26, "signal": 9},
            type="momentum",
        ),
        StrategyInfo(
            name="alpha_ml",
            description="ML-based strategy using LightGBM and XGBoost ensemble",
            parameters={"use_lightgbm": True, "use_xgboost": True},
            type="ml",
        ),
    ]
    return strategies


@app.get("/strategies/{strategy_name}", tags=["Strategies"])
async def get_strategy(strategy_name: str):
    """Get strategy details."""
    try:
        from strategies import create_strategy
        strategy = create_strategy(strategy_name, ["AAPL"])
        
        return {
            "name": strategy.name,
            "description": strategy.description,
            "parameters": strategy.parameters,
            "config": strategy.config.__dict__ if hasattr(strategy, 'config') else {},
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Strategy not found: {e}")


# =============================================================================
# DATA ENDPOINTS
# =============================================================================

@app.get("/data/symbols", tags=["Data"])
async def list_symbols():
    """List available symbols."""
    from config.settings import get_settings
    from data.loader import CSVLoader
    
    settings = get_settings()
    loader = CSVLoader(storage_path=settings.data.storage_path)
    
    return {"symbols": loader.get_available_symbols()}


@app.post("/data", tags=["Data"])
async def get_data(request: DataRequest):
    """Get OHLCV data for a symbol."""
    try:
        from config.settings import get_settings
        from data.loader import CSVLoader
        from data.processor import DataProcessor
        
        settings = get_settings()
        loader = CSVLoader(storage_path=settings.data.storage_path)
        processor = DataProcessor()
        
        df = loader.load(
            request.symbol,
            start_date=datetime.strptime(request.start_date, "%Y-%m-%d") if request.start_date else None,
            end_date=datetime.strptime(request.end_date, "%Y-%m-%d") if request.end_date else None,
            timeframe=request.timeframe,
        )
        
        df = processor.process(df)
        
        # Convert to JSON-friendly format
        data = df.to_dicts()
        for row in data:
            if "timestamp" in row:
                row["timestamp"] = str(row["timestamp"])
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "count": len(data),
            "data": data[:1000],  # Limit response size
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/data/{symbol}/latest", tags=["Data"])
async def get_latest_price(symbol: str):
    """Get latest price for a symbol."""
    try:
        from config.settings import get_settings
        from data.loader import CSVLoader
        
        settings = get_settings()
        loader = CSVLoader(storage_path=settings.data.storage_path)
        
        df = loader.load(symbol)
        latest = df.tail(1).to_dicts()[0]
        
        return {
            "symbol": symbol,
            "timestamp": str(latest.get("timestamp")),
            "open": latest.get("open"),
            "high": latest.get("high"),
            "low": latest.get("low"),
            "close": latest.get("close"),
            "volume": latest.get("volume"),
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# =============================================================================
# TASK ENDPOINTS
# =============================================================================

@app.get("/tasks", tags=["Tasks"])
async def list_tasks():
    """List all tasks."""
    return {
        "tasks": [
            {
                "task_id": task_id,
                "type": task["type"],
                "status": task["status"],
                "created_at": task["created_at"],
            }
            for task_id, task in state.tasks.items()
        ]
    }


@app.get("/tasks/{task_id}", tags=["Tasks"])
async def get_task(task_id: str):
    """Get task details."""
    if task_id not in state.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return state.tasks[task_id]


@app.delete("/tasks/{task_id}", tags=["Tasks"])
async def delete_task(task_id: str):
    """Delete a task."""
    if task_id not in state.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    del state.tasks[task_id]
    return {"message": f"Task {task_id} deleted"}


# =============================================================================
# WEBSOCKET ENDPOINTS
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    state.websockets.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Handle incoming messages
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "subscribe":
                    await websocket.send_json({
                        "type": "subscribed",
                        "channel": message.get("channel"),
                    })
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        state.websockets.remove(websocket)


async def broadcast_task_update(task_id: str):
    """Broadcast task update to all connected clients."""
    if task_id in state.tasks:
        message = {
            "type": "task_update",
            "task_id": task_id,
            "status": state.tasks[task_id]["status"],
            "result": state.tasks[task_id].get("result"),
        }
        
        for websocket in state.websockets:
            try:
                await websocket.send_json(message)
            except Exception:
                pass


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)