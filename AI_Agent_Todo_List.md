# AlphaTrade System - AI Agent Development Task List

## 📋 Complete Implementation Guide for Institutional-Grade Trading Platform

**Document Purpose:** Step-by-step task list for AI Agent to implement all required improvements  
**Total Estimated Duration:** 22 weeks  
**Priority Levels:** 🔴 Critical | 🟠 High | 🟡 Medium | 🟢 Low

---

# PHASE 1: FOUNDATION INFRASTRUCTURE (Weeks 1-4)

## 1.1 Time-Series Database Setup (TimescaleDB)

### Task 1.1.1: Install and Configure TimescaleDB
**Priority:** 🔴 Critical  
**Estimated Time:** 2 days  
**Dependencies:** None

- [ ] Install PostgreSQL 14+ on the server
- [ ] Install TimescaleDB extension
- [ ] Configure postgresql.conf for time-series workloads:
  ```
  shared_preload_libraries = 'timescaledb'
  max_connections = 100
  shared_buffers = 4GB
  effective_cache_size = 12GB
  maintenance_work_mem = 1GB
  work_mem = 256MB
  ```
- [ ] Create database `alphatrade_db`
- [ ] Enable TimescaleDB extension: `CREATE EXTENSION IF NOT EXISTS timescaledb;`
- [ ] Create connection pool configuration (PgBouncer recommended)
- [ ] Set up database user with appropriate permissions
- [ ] Test connection from Python using `psycopg2` or `asyncpg`

### Task 1.1.2: Design Database Schema
**Priority:** 🔴 Critical  
**Estimated Time:** 3 days  
**Dependencies:** Task 1.1.1

- [ ] Create tick data table:
  ```sql
  CREATE TABLE tick_data (
      time TIMESTAMPTZ NOT NULL,
      symbol VARCHAR(20) NOT NULL,
      price DOUBLE PRECISION NOT NULL,
      size BIGINT NOT NULL,
      exchange VARCHAR(10),
      conditions VARCHAR(50)
  );
  SELECT create_hypertable('tick_data', 'time', 
      partitioning_column => 'symbol',
      number_partitions => 4);
  ```
- [ ] Create OHLCV table:
  ```sql
  CREATE TABLE ohlcv_bars (
      time TIMESTAMPTZ NOT NULL,
      symbol VARCHAR(20) NOT NULL,
      timeframe VARCHAR(10) NOT NULL,
      open DOUBLE PRECISION,
      high DOUBLE PRECISION,
      low DOUBLE PRECISION,
      close DOUBLE PRECISION,
      volume BIGINT,
      vwap DOUBLE PRECISION,
      trade_count INTEGER
  );
  SELECT create_hypertable('ohlcv_bars', 'time');
  ```
- [ ] Create continuous aggregates for automatic OHLCV rollups:
  ```sql
  CREATE MATERIALIZED VIEW ohlcv_15min
  WITH (timescaledb.continuous) AS
  SELECT time_bucket('15 minutes', time) AS bucket,
         symbol,
         first(price, time) AS open,
         max(price) AS high,
         min(price) AS low,
         last(price, time) AS close,
         sum(size) AS volume
  FROM tick_data
  GROUP BY bucket, symbol;
  ```
- [ ] Create indexes for common query patterns
- [ ] Set up retention policies (e.g., raw ticks: 30 days, 15min bars: 5 years)
- [ ] Enable compression for historical data
- [ ] Create corporate actions table (splits, dividends)
- [ ] Create metadata table for symbol information

### Task 1.1.3: Build Database Access Layer
**Priority:** 🔴 Critical  
**Estimated Time:** 3 days  
**Dependencies:** Task 1.1.2

- [ ] Create new file: `src/data/storage/timescale_client.py`
- [ ] Implement `TimescaleClient` class with:
  - [ ] Connection pool management
  - [ ] `insert_ticks(ticks: List[Tick])` method
  - [ ] `insert_ohlcv(bars: List[Bar])` method
  - [ ] `get_ohlcv(symbol, start, end, timeframe)` method
  - [ ] `get_ticks(symbol, start, end)` method
  - [ ] Batch insert optimization (COPY command)
  - [ ] Query result caching with Redis (optional)
- [ ] Create async version using `asyncpg`
- [ ] Add connection retry logic with exponential backoff
- [ ] Implement query timeout handling
- [ ] Write unit tests for all database operations
- [ ] Add integration tests with test database

### Task 1.1.4: Data Migration
**Priority:** 🔴 Critical  
**Estimated Time:** 2 days  
**Dependencies:** Task 1.1.3

- [ ] Create migration script: `scripts/migrate_csv_to_timescale.py`
- [ ] Read all existing CSV files from `data/raw/`
- [ ] Validate data using existing `DataValidator`
- [ ] Transform to TimescaleDB format
- [ ] Batch insert with progress tracking
- [ ] Verify row counts match
- [ ] Create checksum validation for data integrity
- [ ] Document migration process
- [ ] Update `DataLoader` to support TimescaleDB as data source

---

## 1.2 Polars Migration (Performance Upgrade)

### Task 1.2.1: Install and Configure Polars
**Priority:** 🟠 High  
**Estimated Time:** 1 day  
**Dependencies:** None

- [ ] Add `polars` to `pyproject.toml` dependencies
- [ ] Install: `pip install polars`
- [ ] Install `pyarrow` for Parquet support
- [ ] Create utility module: `src/utils/polars_utils.py`
- [ ] Add pandas-polars conversion helpers
- [ ] Set up lazy evaluation defaults

### Task 1.2.2: Migrate DataLoader to Polars
**Priority:** 🟠 High  
**Estimated Time:** 3 days  
**Dependencies:** Task 1.2.1

- [ ] Create new file: `src/data/loaders/polars_loader.py`
- [ ] Implement `PolarsDataLoader` class:
  ```python
  class PolarsDataLoader:
      def load_csv(self, path: Path) -> pl.LazyFrame
      def load_parquet(self, path: Path) -> pl.LazyFrame
      def load_from_timescale(self, query: str) -> pl.DataFrame
  ```
- [ ] Use lazy evaluation for memory efficiency
- [ ] Implement parallel file reading with `pl.scan_csv`
- [ ] Add schema validation
- [ ] Maintain backward compatibility with pandas API
- [ ] Benchmark performance vs pandas (document results)
- [ ] Update existing tests

### Task 1.2.3: Migrate Feature Pipeline to Polars
**Priority:** 🟠 High  
**Estimated Time:** 4 days  
**Dependencies:** Task 1.2.2

- [ ] Create `src/features/polars_indicators.py`
- [ ] Reimplement key technical indicators in Polars:
  - [ ] SMA, EMA, WMA (use `rolling_mean`, `ewm_mean`)
  - [ ] RSI
  - [ ] MACD
  - [ ] Bollinger Bands
  - [ ] ATR
  - [ ] Volume indicators
- [ ] Use Polars expressions for vectorized operations
- [ ] Implement window functions with `over()` for cross-sectional features
- [ ] Create `PolarsFeaturePipeline` class
- [ ] Add lazy evaluation support for large datasets
- [ ] Benchmark feature generation speed improvement
- [ ] Write comprehensive tests

---

## 1.3 MLflow Integration (Experiment Tracking)

### Task 1.3.1: Deploy MLflow Server
**Priority:** 🟠 High  
**Estimated Time:** 2 days  
**Dependencies:** None

- [ ] Install MLflow: `pip install mlflow`
- [ ] Set up MLflow tracking server:
  ```bash
  mlflow server \
      --backend-store-uri postgresql://user:pass@localhost/mlflow \
      --default-artifact-root ./mlartifacts \
      --host 0.0.0.0 \
      --port 5000
  ```
- [ ] Configure artifact storage (local or S3)
- [ ] Create MLflow database schema
- [ ] Set up authentication (if needed)
- [ ] Create Docker Compose file for MLflow deployment
- [ ] Test server accessibility

### Task 1.3.2: Integrate MLflow into Training
**Priority:** 🟠 High  
**Estimated Time:** 2 days  
**Dependencies:** Task 1.3.1

- [ ] Create `src/training/experiment_tracker.py`:
  ```python
  class ExperimentTracker:
      def __init__(self, experiment_name: str)
      def start_run(self, run_name: str)
      def log_params(self, params: dict)
      def log_metrics(self, metrics: dict, step: int = None)
      def log_model(self, model, artifact_path: str)
      def log_artifact(self, local_path: str)
      def end_run()
  ```
- [ ] Add MLflow tracking to `MLAlphaStrategy.fit()`:
  - [ ] Log all hyperparameters
  - [ ] Log training metrics per epoch/iteration
  - [ ] Log validation metrics
  - [ ] Log feature importance
  - [ ] Log model artifacts
- [ ] Create model signature for input/output schema
- [ ] Set up automatic experiment naming convention
- [ ] Add tags for easy filtering (strategy type, symbol, timeframe)

### Task 1.3.3: Implement Model Registry
**Priority:** 🟠 High  
**Estimated Time:** 2 days  
**Dependencies:** Task 1.3.2

- [ ] Create `src/training/model_registry.py`:
  ```python
  class ModelRegistry:
      def register_model(self, run_id: str, model_name: str)
      def transition_stage(self, model_name: str, version: int, stage: str)
      def get_latest_model(self, model_name: str, stage: str)
      def load_model(self, model_uri: str)
      def compare_models(self, model_name: str, versions: List[int])
  ```
- [ ] Implement staging workflow: None → Staging → Production → Archived
- [ ] Add model versioning with semantic versioning
- [ ] Create model comparison utilities
- [ ] Add model deletion/archival policies
- [ ] Document model registry workflow

---

## 1.4 Configuration & Code Quality

### Task 1.4.1: Externalize Hardcoded Parameters
**Priority:** 🟡 Medium  
**Estimated Time:** 2 days  
**Dependencies:** None

- [ ] Audit codebase for hardcoded values:
  - [ ] `random_state=42` → make configurable for ensemble diversity
  - [ ] `periods_per_year` (252*26) → move to config
  - [ ] Feature selection limit (50) → move to config
  - [ ] Default model parameters → move to YAML
- [ ] Update `config/settings.py` with new parameters
- [ ] Create `config/ml_config.yaml`:
  ```yaml
  random_seed: 42
  periods_per_year:
    daily: 252
    hourly: 1638  # 252 * 6.5
    15min: 6552   # 252 * 26
  feature_selection:
    max_features: 50
    method: "importance"
  ```
- [ ] Update all affected modules to read from config
- [ ] Add environment variable overrides
- [ ] Document all configuration options

### Task 1.4.2: Add Type Hints and Documentation
**Priority:** 🟢 Low  
**Estimated Time:** 2 days  
**Dependencies:** None

- [ ] Run `mypy` on entire codebase, fix type errors
- [ ] Add missing type hints to all public functions
- [ ] Add docstrings following Google style guide
- [ ] Generate API documentation with Sphinx
- [ ] Create architecture decision records (ADRs)
- [ ] Update README with new setup instructions

---

# PHASE 2: ML TRAINING PIPELINE (Weeks 5-10)

## 2.1 Create Dedicated Training Module

### Task 2.1.1: Design Training Module Architecture
**Priority:** 🔴 Critical  
**Estimated Time:** 1 day  
**Dependencies:** Phase 1 complete

- [ ] Create directory structure:
  ```
  src/training/
  ├── __init__.py
  ├── trainer.py           # Main training orchestrator
  ├── model_factory.py     # Model creation
  ├── validation.py        # Cross-validation strategies
  ├── optimization.py      # Hyperparameter optimization
  ├── callbacks.py         # Training callbacks
  ├── losses.py            # Custom loss functions
  ├── metrics.py           # Financial metrics for training
  └── experiment_tracker.py # MLflow integration
  ```
- [ ] Define interfaces and abstract base classes
- [ ] Create sequence diagrams for training workflow
- [ ] Document API contracts

### Task 2.1.2: Implement ModelFactory
**Priority:** 🔴 Critical  
**Estimated Time:** 3 days  
**Dependencies:** Task 2.1.1

- [ ] Create `src/training/model_factory.py`:
  ```python
  class ModelFactory:
      @staticmethod
      def create_model(model_type: str, params: dict) -> BaseModel:
          """Factory method for model creation"""
      
      @staticmethod
      def get_default_params(model_type: str) -> dict:
          """Return default hyperparameters"""
      
      @staticmethod
      def get_param_space(model_type: str) -> dict:
          """Return Optuna parameter space"""
  ```
- [ ] Implement model types:
  - [ ] `lightgbm_classifier` / `lightgbm_regressor`
  - [ ] `xgboost_classifier` / `xgboost_regressor`
  - [ ] `catboost_classifier` / `catboost_regressor`
  - [ ] `random_forest`
  - [ ] `linear_model` (Ridge, Lasso, ElasticNet)
- [ ] Add GPU support detection and configuration
- [ ] Implement model serialization/deserialization
- [ ] Add model warm-starting capability
- [ ] Write factory tests

### Task 2.1.3: Implement Main Trainer Class
**Priority:** 🔴 Critical  
**Estimated Time:** 4 days  
**Dependencies:** Task 2.1.2

- [ ] Create `src/training/trainer.py`:
  ```python
  class Trainer:
      def __init__(
          self,
          model_factory: ModelFactory,
          validator: ValidationStrategy,
          tracker: ExperimentTracker,
          callbacks: List[Callback] = None
      )
      
      def fit(
          self,
          X: pd.DataFrame,
          y: pd.Series,
          sample_weights: pd.Series = None
      ) -> TrainingResult
      
      def predict(self, X: pd.DataFrame) -> np.ndarray
      
      def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict
      
      def save(self, path: str)
      
      @classmethod
      def load(cls, path: str) -> "Trainer"
  ```
- [ ] Implement training loop with:
  - [ ] Early stopping
  - [ ] Learning rate scheduling
  - [ ] Gradient clipping (for neural networks)
  - [ ] Checkpoint saving
- [ ] Add callback system for extensibility
- [ ] Implement training result dataclass
- [ ] Add training progress logging
- [ ] Write comprehensive tests

### Task 2.1.4: Implement Custom Loss Functions
**Priority:** 🟠 High  
**Estimated Time:** 2 days  
**Dependencies:** Task 2.1.3

- [ ] Create `src/training/losses.py`:
  ```python
  def sharpe_loss(y_true, y_pred, returns):
      """Negative Sharpe ratio as loss function"""
      
  def sortino_loss(y_true, y_pred, returns):
      """Negative Sortino ratio as loss function"""
      
  def max_drawdown_penalty(y_true, y_pred, equity):
      """Penalize strategies with high drawdown"""
      
  def profit_factor_loss(y_true, y_pred, pnl):
      """Optimize for profit factor"""
  ```
- [ ] Implement gradient-compatible versions for boosting
- [ ] Add combined multi-objective loss
- [ ] Create loss function registry
- [ ] Document mathematical formulations
- [ ] Test with LightGBM custom objective

---

## 2.2 Purged K-Fold Cross-Validation

### Task 2.2.1: Enhance TimeSeriesSplitter
**Priority:** 🔴 Critical  
**Estimated Time:** 3 days  
**Dependencies:** Task 2.1.1

- [ ] Create `src/training/validation.py`:
  ```python
  class PurgedKFoldCV:
      def __init__(
          self,
          n_splits: int = 5,
          purge_gap: int = None,  # Auto-calculate if None
          embargo_pct: float = 0.01,
          prediction_horizon: int = 1,
          max_feature_lookback: int = 20
      )
      
      def split(self, X, y=None, groups=None):
          """Generate purged train/test indices"""
      
      def get_purge_gap(self) -> int:
          """Calculate optimal purge gap"""
  ```
- [ ] Implement purge gap calculation:
  ```python
  purge_gap = prediction_horizon + max_feature_lookback
  ```
- [ ] Implement embargo period after test set
- [ ] Add visualization of splits
- [ ] Handle overlapping labels correctly
- [ ] Write mathematical documentation
- [ ] Create unit tests with edge cases

### Task 2.2.2: Implement Combinatorial Purged CV
**Priority:** 🟠 High  
**Estimated Time:** 2 days  
**Dependencies:** Task 2.2.1

- [ ] Create `CombinatorialPurgedKFoldCV` class:
  ```python
  class CombinatorialPurgedKFoldCV:
      """
      Combinatorial Purged Cross-Validation
      Tests on multiple non-overlapping paths through time
      """
      def __init__(
          self,
          n_splits: int = 5,
          n_test_splits: int = 2,
          purge_gap: int = None,
          embargo_pct: float = 0.01
      )
  ```
- [ ] Generate all valid train/test combinations
- [ ] Ensure no overlap between test periods
- [ ] Calculate effective number of paths
- [ ] Add path visualization
- [ ] Document statistical properties

### Task 2.2.3: Implement Walk-Forward Validation
**Priority:** 🟠 High  
**Estimated Time:** 2 days  
**Dependencies:** Task 2.2.1

- [ ] Enhance existing `WalkForwardOptimizer`:
  ```python
  class WalkForwardValidator:
      def __init__(
          self,
          train_period: int,
          test_period: int,
          step_size: int = None,  # Default: test_period
          expanding: bool = False,
          purge_gap: int = 10,
          embargo_bars: int = 5
      )
      
      def split(self, X, y=None) -> Generator[Tuple[np.ndarray, np.ndarray]]
      
      def get_n_splits(self, X) -> int
  ```
- [ ] Support both expanding and sliding window
- [ ] Add gap between train end and test start
- [ ] Implement result aggregation across folds
- [ ] Create walk-forward visualization
- [ ] Add statistical significance tests

---

## 2.3 Hyperparameter Optimization (Optuna)

### Task 2.3.1: Install and Configure Optuna
**Priority:** 🔴 Critical  
**Estimated Time:** 1 day  
**Dependencies:** Task 2.1.3

- [ ] Add to dependencies: `pip install optuna optuna-dashboard`
- [ ] Set up Optuna storage backend (SQLite or PostgreSQL)
- [ ] Configure Optuna dashboard for visualization
- [ ] Create `config/optuna_config.yaml`:
  ```yaml
  study:
    direction: maximize  # Sharpe ratio
    sampler: TPESampler
    pruner: MedianPruner
  optimization:
    n_trials: 100
    timeout: 3600  # 1 hour
    n_jobs: -1  # Parallel
  ```

### Task 2.3.2: Implement OptunaOptimizer
**Priority:** 🔴 Critical  
**Estimated Time:** 4 days  
**Dependencies:** Task 2.3.1

- [ ] Create `src/training/optimization.py`:
  ```python
  class OptunaOptimizer:
      def __init__(
          self,
          model_type: str,
          validation_strategy: ValidationStrategy,
          objective_metric: str = "sharpe_ratio",
          direction: str = "maximize",
          n_trials: int = 100,
          timeout: int = None,
          pruner: optuna.pruners.BasePruner = None,
          sampler: optuna.samplers.BaseSampler = None
      )
      
      def optimize(
          self,
          X: pd.DataFrame,
          y: pd.Series,
          param_space: dict = None
      ) -> OptimizationResult
      
      def _objective(self, trial: optuna.Trial) -> float:
          """Objective function for Optuna"""
      
      def get_best_params(self) -> dict
      
      def get_param_importance(self) -> pd.DataFrame
  ```
- [ ] Define parameter spaces for each model type:
  ```python
  LIGHTGBM_PARAM_SPACE = {
      "n_estimators": ("int", 50, 500),
      "max_depth": ("int", 3, 12),
      "learning_rate": ("float_log", 0.01, 0.3),
      "num_leaves": ("int", 20, 100),
      "min_child_samples": ("int", 10, 100),
      "subsample": ("float", 0.6, 1.0),
      "colsample_bytree": ("float", 0.6, 1.0),
      "reg_alpha": ("float_log", 1e-8, 10.0),
      "reg_lambda": ("float_log", 1e-8, 10.0),
  }
  ```
- [ ] Implement early stopping callback for pruning
- [ ] Add cross-validation within objective
- [ ] Log all trials to MLflow
- [ ] Create optimization visualization
- [ ] Implement warm-starting from previous study

### Task 2.3.3: Implement Multi-Objective Optimization
**Priority:** 🟡 Medium  
**Estimated Time:** 2 days  
**Dependencies:** Task 2.3.2

- [ ] Create `MultiObjectiveOptimizer` class:
  ```python
  class MultiObjectiveOptimizer(OptunaOptimizer):
      def __init__(
          self,
          objectives: List[str] = ["sharpe_ratio", "max_drawdown"],
          directions: List[str] = ["maximize", "minimize"]
      )
      
      def get_pareto_front(self) -> List[OptimizationResult]
  ```
- [ ] Use NSGA-II sampler for multi-objective
- [ ] Visualize Pareto front
- [ ] Implement selection strategy for best trade-off
- [ ] Add constraint handling (e.g., min Sharpe > 1.0)

---

## 2.4 Deep Learning Models

### Task 2.4.1: Set Up PyTorch Lightning
**Priority:** 🟠 High  
**Estimated Time:** 2 days  
**Dependencies:** Task 2.1.3

- [ ] Add dependencies:
  ```
  torch>=2.0.0
  pytorch-lightning>=2.0.0
  torchmetrics>=1.0.0
  ```
- [ ] Create `src/training/deep_learning/__init__.py`
- [ ] Set up GPU/MPS detection and configuration
- [ ] Create base `LightningModule` for financial models
- [ ] Implement data module for time-series batching
- [ ] Configure logging integration with MLflow

### Task 2.4.2: Implement LSTM Model
**Priority:** 🟠 High  
**Estimated Time:** 3 days  
**Dependencies:** Task 2.4.1

- [ ] Create `src/training/deep_learning/lstm.py`:
  ```python
  class LSTMPredictor(pl.LightningModule):
      def __init__(
          self,
          input_size: int,
          hidden_size: int = 128,
          num_layers: int = 2,
          dropout: float = 0.2,
          bidirectional: bool = False,
          output_size: int = 1,
          learning_rate: float = 1e-3
      )
      
      def forward(self, x: torch.Tensor) -> torch.Tensor
      
      def training_step(self, batch, batch_idx) -> torch.Tensor
      
      def validation_step(self, batch, batch_idx)
      
      def configure_optimizers(self)
  ```
- [ ] Implement attention mechanism:
  ```python
  class AttentionLSTM(LSTMPredictor):
      """LSTM with self-attention layer"""
  ```
- [ ] Add layer normalization
- [ ] Implement sequence padding/masking
- [ ] Create time-series dataset class
- [ ] Add learning rate finder
- [ ] Write model tests

### Task 2.4.3: Implement Transformer Model
**Priority:** 🟠 High  
**Estimated Time:** 4 days  
**Dependencies:** Task 2.4.1

- [ ] Create `src/training/deep_learning/transformer.py`:
  ```python
  class TemporalFusionTransformer(pl.LightningModule):
      """
      Temporal Fusion Transformer for financial time-series
      Based on: https://arxiv.org/abs/1912.09363
      """
      def __init__(
          self,
          input_size: int,
          hidden_size: int = 160,
          num_attention_heads: int = 4,
          num_encoder_layers: int = 2,
          dropout: float = 0.1,
          static_features: int = 0,
          prediction_horizon: int = 1
      )
  ```
- [ ] Implement positional encoding for time-series
- [ ] Add variable selection network
- [ ] Implement gated residual network
- [ ] Add interpretable multi-head attention
- [ ] Create quantile output for uncertainty estimation
- [ ] Implement entity embedding for categorical features
- [ ] Add gradient clipping
- [ ] Write comprehensive tests

### Task 2.4.4: Create Deep Learning Trainer Integration
**Priority:** 🟠 High  
**Estimated Time:** 2 days  
**Dependencies:** Tasks 2.4.2, 2.4.3

- [ ] Create `DeepLearningTrainer` wrapper:
  ```python
  class DeepLearningTrainer(Trainer):
      def __init__(
          self,
          model_class: Type[pl.LightningModule],
          model_params: dict,
          trainer_params: dict = None
      )
      
      def fit(self, X, y, val_X=None, val_y=None)
      
      def predict(self, X) -> np.ndarray
  ```
- [ ] Integrate with Optuna for hyperparameter tuning
- [ ] Add checkpoint saving/loading
- [ ] Implement early stopping with patience
- [ ] Add learning rate scheduling (OneCycleLR, ReduceLROnPlateau)
- [ ] Create TensorBoard logging
- [ ] Integrate with MLflow artifact logging

---

# PHASE 3: ADVANCED FEATURE ENGINEERING (Weeks 11-16)

## 3.1 Fractional Differentiation

### Task 3.1.1: Implement Fractional Differentiation
**Priority:** 🔴 Critical  
**Estimated Time:** 3 days  
**Dependencies:** Phase 2 complete

- [ ] Create `src/features/fractional_diff.py`:
  ```python
  def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
      """
      Calculate weights for fixed-width window fractional differentiation
      Based on: Advances in Financial Machine Learning, Ch. 5
      """
  
  def frac_diff_ffd(
      series: pd.Series,
      d: float,
      threshold: float = 1e-5
  ) -> pd.Series:
      """Apply fractional differentiation with fixed window"""
  
  def find_min_d(
      series: pd.Series,
      d_range: Tuple[float, float] = (0, 1),
      p_value_threshold: float = 0.05
  ) -> float:
      """Find minimum d that achieves stationarity (ADF test)"""
  ```
- [ ] Implement ADF test integration
- [ ] Add KPSS test as alternative
- [ ] Create auto-differentiation for each feature
- [ ] Visualize d vs. ADF p-value and correlation with original
- [ ] Handle edge cases (constant series, NaN values)
- [ ] Write mathematical documentation
- [ ] Create comprehensive unit tests

### Task 3.1.2: Integrate into Feature Pipeline
**Priority:** 🔴 Critical  
**Estimated Time:** 2 days  
**Dependencies:** Task 3.1.1

- [ ] Add `FractionalDiffTransformer` to pipeline:
  ```python
  class FractionalDiffTransformer(BaseEstimator, TransformerMixin):
      def __init__(
          self,
          d: float = None,  # Auto-find if None
          columns: List[str] = None,
          threshold: float = 1e-5
      )
      
      def fit(self, X, y=None):
          """Find optimal d for each column"""
      
      def transform(self, X) -> pd.DataFrame:
          """Apply fractional differentiation"""
  ```
- [ ] Store optimal d values per feature
- [ ] Add to `FeaturePipeline` configuration
- [ ] Update feature configuration YAML
- [ ] Benchmark impact on model performance

---

## 3.2 Order Book Features

### Task 3.2.1: Create Order Book Data Structures
**Priority:** 🟠 High  
**Estimated Time:** 2 days  
**Dependencies:** Task 1.1.2

- [ ] Create `src/data/order_book.py`:
  ```python
  @dataclass
  class OrderBookLevel:
      price: float
      size: int
      order_count: int
  
  @dataclass
  class OrderBookSnapshot:
      timestamp: datetime
      symbol: str
      bids: List[OrderBookLevel]  # Best to worst
      asks: List[OrderBookLevel]  # Best to worst
  
  class OrderBook:
      def update(self, message: dict)
      def get_snapshot(self, levels: int = 10) -> OrderBookSnapshot
      def get_mid_price(self) -> float
      def get_spread(self) -> float
      def get_imbalance(self, levels: int = 5) -> float
  ```
- [ ] Implement order book reconstruction from L2 data
- [ ] Add efficient update mechanism
- [ ] Handle crossed books
- [ ] Add serialization for storage

### Task 3.2.2: Implement Order Book Features
**Priority:** 🟠 High  
**Estimated Time:** 3 days  
**Dependencies:** Task 3.2.1

- [ ] Create `src/features/orderbook_features.py`:
  ```python
  class OrderBookFeatures:
      def bid_ask_imbalance(self, levels: int = 5) -> float:
          """(Bid Vol - Ask Vol) / (Bid Vol + Ask Vol)"""
      
      def weighted_mid_price(self) -> float:
          """Volume-weighted mid price"""
      
      def order_flow_imbalance(self, window: int = 100) -> float:
          """Cumulative signed trade flow"""
      
      def depth_ratio(self, levels: int = 5) -> float:
          """Total bid depth / Total ask depth"""
      
      def spread_bps(self) -> float:
          """Spread in basis points"""
      
      def microprice(self) -> float:
          """Size-weighted mid price"""
      
      def book_pressure(self, decay: float = 0.5) -> float:
          """Exponentially weighted book pressure"""
      
      def trade_arrival_intensity(self, window: int = 60) -> float:
          """Trades per second (Hawkes process)"""
  ```
- [ ] Add rolling/expanding versions of all features
- [ ] Implement cross-sectional ranking
- [ ] Handle missing/stale data
- [ ] Create visualization tools
- [ ] Write feature documentation

---

## 3.3 Statistical Arbitrage Features

### Task 3.3.1: Implement Cointegration Testing
**Priority:** 🟠 High  
**Estimated Time:** 3 days  
**Dependencies:** Task 3.1.1

- [ ] Create `src/features/cointegration.py`:
  ```python
  class CointegrationAnalyzer:
      def engle_granger_test(
          self,
          series1: pd.Series,
          series2: pd.Series
      ) -> Tuple[float, float, bool]:
          """Returns: (test_stat, p_value, is_cointegrated)"""
      
      def johansen_test(
          self,
          data: pd.DataFrame,
          det_order: int = 0
      ) -> JohansenResult:
          """Multi-variate cointegration test"""
      
      def find_cointegrated_pairs(
          self,
          returns: pd.DataFrame,
          p_threshold: float = 0.05
      ) -> List[Tuple[str, str, float]]:
          """Find all cointegrated pairs in universe"""
      
      def calculate_spread(
          self,
          series1: pd.Series,
          series2: pd.Series,
          hedge_ratio: float = None
      ) -> pd.Series:
          """Calculate spread with optimal hedge ratio"""
      
      def half_life(self, spread: pd.Series) -> float:
          """Estimate mean-reversion half-life"""
  ```
- [ ] Implement rolling cointegration test
- [ ] Add hedge ratio estimation (OLS, TLS, Kalman)
- [ ] Create pair selection framework
- [ ] Add visualization for spread and z-score
- [ ] Document statistical methodology

### Task 3.3.2: Implement Ornstein-Uhlenbeck Features
**Priority:** 🟠 High  
**Estimated Time:** 2 days  
**Dependencies:** Task 3.3.1

- [ ] Create `OrnsteinUhlenbeckEstimator`:
  ```python
  class OrnsteinUhlenbeckEstimator:
      def fit(self, series: pd.Series) -> OUParams:
          """
          Estimate OU parameters: dX = θ(μ - X)dt + σdW
          Returns: theta (speed), mu (mean), sigma (volatility)
          """
      
      def half_life(self) -> float:
          """Mean-reversion half-life: ln(2) / theta"""
      
      def expected_value(self, x0: float, t: float) -> float:
          """E[X_t | X_0 = x0]"""
      
      def variance(self, t: float) -> float:
          """Var[X_t]"""
  ```
- [ ] Add rolling parameter estimation
- [ ] Create confidence intervals
- [ ] Implement regime detection for OU parameters

---

## 3.4 Macroeconomic Features

### Task 3.4.1: FRED Data Integration
**Priority:** 🟡 Medium  
**Estimated Time:** 2 days  
**Dependencies:** None

- [ ] Install `fredapi`: `pip install fredapi`
- [ ] Create `src/data/macro/fred_client.py`:
  ```python
  class FREDClient:
      def __init__(self, api_key: str)
      
      def get_series(
          self,
          series_id: str,
          start_date: str = None,
          end_date: str = None
      ) -> pd.Series
      
      def get_yield_curve(self, date: str = None) -> pd.Series
      
      def get_economic_indicators(self) -> pd.DataFrame
  ```
- [ ] Implement key series fetching:
  - [ ] Treasury yields (DGS1, DGS2, DGS5, DGS10, DGS30)
  - [ ] Fed Funds Rate (FEDFUNDS)
  - [ ] VIX (VIXCLS)
  - [ ] Credit spreads (BAMLC0A4CBBB, BAMLH0A0HYM2)
  - [ ] Unemployment (UNRATE)
  - [ ] CPI (CPIAUCSL)
- [ ] Add caching for API calls
- [ ] Handle data frequency alignment (daily, weekly, monthly)

### Task 3.4.2: Implement Macro Features
**Priority:** 🟡 Medium  
**Estimated Time:** 2 days  
**Dependencies:** Task 3.4.1

- [ ] Create `src/features/macro_features.py`:
  ```python
  class MacroFeatures:
      def yield_curve_slope(self) -> pd.Series:
          """10Y - 2Y spread"""
      
      def yield_curve_curvature(self) -> pd.Series:
          """2*(5Y) - (2Y + 10Y)"""
      
      def credit_spread(self) -> pd.Series:
          """BBB - Treasury spread"""
      
      def real_rate(self) -> pd.Series:
          """Nominal rate - inflation expectations"""
      
      def economic_surprise(self) -> pd.Series:
          """Actual vs consensus economic releases"""
      
      def risk_appetite(self) -> pd.Series:
          """Composite risk-on/risk-off indicator"""
  ```
- [ ] Handle data publication lags (point-in-time correctness)
- [ ] Implement forward-fill for lower frequency data
- [ ] Add regime indicators based on macro state
- [ ] Create macro dashboard visualization

---

## 3.5 Feature Store Implementation

### Task 3.5.1: Deploy Feast Feature Store
**Priority:** 🟡 Medium  
**Estimated Time:** 3 days  
**Dependencies:** Tasks 3.1-3.4

- [ ] Install Feast: `pip install feast`
- [ ] Create feature repository structure:
  ```
  feature_repo/
  ├── feature_store.yaml
  ├── features/
  │   ├── technical_features.py
  │   ├── orderbook_features.py
  │   ├── macro_features.py
  │   └── derived_features.py
  └── data_sources.py
  ```
- [ ] Define feature views:
  ```python
  technical_features = FeatureView(
      name="technical_features",
      entities=["symbol"],
      ttl=timedelta(days=1),
      features=[
          Feature(name="rsi_14", dtype=Float32),
          Feature(name="macd_hist", dtype=Float32),
          # ... more features
      ],
      online=True,
      source=technical_source,
  )
  ```
- [ ] Configure offline store (Parquet or BigQuery)
- [ ] Configure online store (Redis or DynamoDB)
- [ ] Set up materialization jobs
- [ ] Create feature serving endpoint

### Task 3.5.2: Integrate Feature Store with Training
**Priority:** 🟡 Medium  
**Estimated Time:** 2 days  
**Dependencies:** Task 3.5.1

- [ ] Create `src/features/feature_store_client.py`:
  ```python
  class FeatureStoreClient:
      def get_historical_features(
          self,
          entity_df: pd.DataFrame,  # timestamp, symbol
          feature_views: List[str]
      ) -> pd.DataFrame:
          """Point-in-time correct feature retrieval"""
      
      def get_online_features(
          self,
          entity_rows: List[dict]
      ) -> dict:
          """Real-time feature serving"""
  ```
- [ ] Ensure point-in-time correctness in training
- [ ] Add feature versioning
- [ ] Create feature documentation generator
- [ ] Add feature monitoring and drift detection

---

# PHASE 4: INSTITUTIONAL BACKTESTING (Weeks 17-22)

## 4.1 Event-Driven Architecture

### Task 4.1.1: Design Event System
**Priority:** 🔴 Critical  
**Estimated Time:** 2 days  
**Dependencies:** Phase 3 complete

- [ ] Create `src/backtesting/events/`:
  ```
  events/
  ├── __init__.py
  ├── base.py        # Event base class
  ├── market.py      # Market data events
  ├── signal.py      # Signal events
  ├── order.py       # Order events
  ├── fill.py        # Fill events
  └── queue.py       # Event queue
  ```
- [ ] Define event hierarchy:
  ```python
  @dataclass
  class Event:
      timestamp: datetime
      event_type: EventType
  
  @dataclass
  class MarketEvent(Event):
      symbol: str
      data: dict  # OHLCV or tick
  
  @dataclass
  class SignalEvent(Event):
      symbol: str
      signal_type: SignalType  # LONG, SHORT, EXIT
      strength: float
      
  @dataclass
  class OrderEvent(Event):
      symbol: str
      order_type: OrderType
      quantity: float
      direction: Direction
      limit_price: float = None
  
  @dataclass
  class FillEvent(Event):
      symbol: str
      quantity: float
      fill_price: float
      commission: float
      slippage: float
  ```
- [ ] Implement priority queue for events
- [ ] Add event logging and replay capability

### Task 4.1.2: Implement Event-Driven Engine
**Priority:** 🔴 Critical  
**Estimated Time:** 5 days  
**Dependencies:** Task 4.1.1

- [ ] Create `src/backtesting/event_engine.py`:
  ```python
  class EventDrivenBacktest:
      def __init__(
          self,
          data_handler: DataHandler,
          strategy: Strategy,
          portfolio: Portfolio,
          execution_handler: ExecutionHandler,
          risk_manager: RiskManager
      )
      
      def run(self) -> BacktestResult:
          """Main event loop"""
          while True:
              event = self.event_queue.get()
              if event is None:
                  break
              self._process_event(event)
      
      def _process_event(self, event: Event):
          if isinstance(event, MarketEvent):
              self.strategy.on_market_data(event)
              self.portfolio.update_positions(event)
          elif isinstance(event, SignalEvent):
              self.portfolio.on_signal(event)
          elif isinstance(event, OrderEvent):
              self.execution_handler.execute(event)
          elif isinstance(event, FillEvent):
              self.portfolio.on_fill(event)
  ```
- [ ] Implement `DataHandler` for bar/tick streaming
- [ ] Create `Portfolio` class with position tracking
- [ ] Implement order management system
- [ ] Add risk checks before order submission
- [ ] Create performance tracking throughout backtest
- [ ] Write integration tests

### Task 4.1.3: Implement Order Book Simulator
**Priority:** 🟠 High  
**Estimated Time:** 3 days  
**Dependencies:** Task 4.1.2

- [ ] Create `src/backtesting/orderbook_sim.py`:
  ```python
  class OrderBookSimulator:
      def __init__(
          self,
          initial_book: OrderBookSnapshot = None,
          tick_size: float = 0.01
      )
      
      def submit_order(self, order: Order) -> Optional[Fill]:
          """Simulate order against book"""
      
      def simulate_market_order(self, order: Order) -> Fill:
          """Walk the book for market orders"""
      
      def simulate_limit_order(self, order: Order) -> Optional[Fill]:
          """Check for immediate fill, else queue"""
      
      def update_book(self, market_data: MarketEvent):
          """Update simulated book state"""
      
      def get_queue_position(self, order: Order) -> int:
          """Estimate position in queue"""
  ```
- [ ] Implement realistic queue priority
- [ ] Add partial fill logic
- [ ] Handle order cancellation
- [ ] Simulate queue jumping on aggressive orders

---

## 4.2 Advanced Transaction Cost Models

### Task 4.2.1: Implement Almgren-Chriss Model
**Priority:** 🔴 Critical  
**Estimated Time:** 3 days  
**Dependencies:** Task 4.1.2

- [ ] Create `src/backtesting/market_impact.py`:
  ```python
  class AlmgrenChrissModel:
      """
      Market impact model based on:
      Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions"
      """
      def __init__(
          self,
          sigma: float,      # Daily volatility
          eta: float,        # Temporary impact coefficient
          gamma: float,      # Permanent impact coefficient
          lambda_: float,    # Risk aversion
          adv: float         # Average daily volume
      )
      
      def temporary_impact(self, rate: float) -> float:
          """Temporary price impact: η * (rate / ADV)"""
      
      def permanent_impact(self, quantity: float) -> float:
          """Permanent price impact: γ * (Q / ADV)"""
      
      def total_cost(
          self,
          quantity: float,
          execution_time: float
      ) -> float:
          """Total expected execution cost"""
      
      def optimal_trajectory(
          self,
          quantity: float,
          time_horizon: float,
          n_steps: int
      ) -> np.ndarray:
          """Optimal execution trajectory"""
  ```
- [ ] Calibrate parameters from historical data
- [ ] Add estimation uncertainty
- [ ] Create visualization of impact curves

### Task 4.2.2: Implement Dynamic Spread Model
**Priority:** 🟠 High  
**Estimated Time:** 2 days  
**Dependencies:** Task 4.2.1

- [ ] Create `src/backtesting/spread_model.py`:
  ```python
  class DynamicSpreadModel:
      def __init__(
          self,
          base_spread_bps: float = 1.0,
          volatility_sensitivity: float = 0.5,
          volume_sensitivity: float = 0.3
      )
      
      def estimate_spread(
          self,
          volatility: float,
          volume_ratio: float,  # Current / Average
          time_of_day: datetime
      ) -> float:
          """Estimate current bid-ask spread"""
      
      def get_effective_price(
          self,
          mid_price: float,
          side: str,  # 'buy' or 'sell'
          size: float
      ) -> float:
          """Price including spread and size impact"""
  ```
- [ ] Add intraday spread patterns
- [ ] Model spread widening during volatility
- [ ] Handle illiquid periods (open, close)

### Task 4.2.3: Implement Latency Simulation
**Priority:** 🟡 Medium  
**Estimated Time:** 2 days  
**Dependencies:** Task 4.1.2

- [ ] Create `src/backtesting/latency.py`:
  ```python
  class LatencySimulator:
      def __init__(
          self,
          signal_to_order_ms: int = 10,
          order_to_exchange_ms: int = 5,
          exchange_processing_ms: int = 1,
          fill_notification_ms: int = 5,
          jitter_pct: float = 0.2
      )
      
      def add_latency(
          self,
          event: Event,
          latency_type: str
      ) -> Event:
          """Add realistic latency to event timestamp"""
      
      def simulate_race_condition(
          self,
          orders: List[Order]
      ) -> List[Order]:
          """Simulate order arrival order uncertainty"""
  ```
- [ ] Model network latency distribution
- [ ] Add burst latency during high activity
- [ ] Simulate order rejection due to stale prices

---

## 4.3 Advanced Performance Metrics

### Task 4.3.1: Implement Additional Metrics
**Priority:** 🔴 Critical  
**Estimated Time:** 2 days  
**Dependencies:** Task 4.1.2

- [ ] Enhance `src/backtesting/metrics.py`:
  ```python
  def omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
      """Probability-weighted ratio of gains vs losses"""
  
  def tail_ratio(returns: pd.Series) -> float:
      """95th percentile gain / 5th percentile loss"""
  
  def gain_to_pain_ratio(returns: pd.Series) -> float:
      """Sum of returns / Sum of absolute losses"""
  
  def ulcer_index(equity: pd.Series) -> float:
      """Root mean square of drawdowns"""
  
  def pain_index(equity: pd.Series) -> float:
      """Mean of drawdowns"""
  
  def burke_ratio(returns: pd.Series, equity: pd.Series) -> float:
      """Excess return / Square root of sum of squared drawdowns"""
  
  def kappa_three(returns: pd.Series, threshold: float = 0) -> float:
      """Generalized Omega with cubic lower partial moment"""
  ```
- [ ] Add confidence intervals for all metrics
- [ ] Implement rolling metric calculation
- [ ] Create metric comparison across strategies

### Task 4.3.2: Implement Probabilistic Sharpe Ratio
**Priority:** 🔴 Critical  
**Estimated Time:** 2 days  
**Dependencies:** Task 4.3.1

- [ ] Create `src/backtesting/statistical_tests.py`:
  ```python
  def probabilistic_sharpe_ratio(
      observed_sharpe: float,
      benchmark_sharpe: float,
      n_observations: int,
      skewness: float,
      kurtosis: float
  ) -> float:
      """
      P(true Sharpe > benchmark) accounting for estimation error
      Based on: Bailey & López de Prado (2012)
      """
  
  def deflated_sharpe_ratio(
      observed_sharpe: float,
      n_trials: int,
      variance_of_trials: float,
      n_observations: int,
      skewness: float,
      kurtosis: float
  ) -> float:
      """
      Sharpe adjusted for multiple testing
      Based on: Bailey & López de Prado (2014)
      """
  
  def minimum_track_record_length(
      observed_sharpe: float,
      benchmark_sharpe: float,
      skewness: float,
      kurtosis: float,
      confidence: float = 0.95
  ) -> int:
      """Minimum observations needed for statistical significance"""
  ```
- [ ] Add strategy comparison tests
- [ ] Implement false discovery rate adjustment
- [ ] Create statistical significance dashboard

### Task 4.3.3: Implement Monte Carlo Analysis
**Priority:** 🟠 High  
**Estimated Time:** 3 days  
**Dependencies:** Task 4.3.1

- [ ] Create `src/backtesting/monte_carlo.py`:
  ```python
  class MonteCarloAnalyzer:
      def __init__(
          self,
          n_simulations: int = 10000,
          confidence_levels: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
      )
      
      def bootstrap_returns(
          self,
          returns: pd.Series,
          block_size: int = 20
      ) -> np.ndarray:
          """Block bootstrap for autocorrelated returns"""
      
      def simulate_paths(
          self,
          returns: pd.Series
      ) -> np.ndarray:
          """Generate simulated equity paths"""
      
      def confidence_intervals(
          self,
          metric_func: Callable,
          returns: pd.Series
      ) -> dict:
          """Calculate confidence intervals for any metric"""
      
      def worst_case_analysis(
          self,
          returns: pd.Series,
          percentile: float = 5
      ) -> dict:
          """Analyze worst-case scenarios"""
      
      def drawdown_distribution(
          self,
          returns: pd.Series
      ) -> pd.DataFrame:
          """Distribution of max drawdowns"""
  ```
- [ ] Implement path-dependent scenario analysis
- [ ] Add regime-aware simulation
- [ ] Create visualization for confidence bands
- [ ] Add stress testing scenarios

---

## 4.4 Reporting and Visualization

### Task 4.4.1: Create Comprehensive Report Generator
**Priority:** 🟡 Medium  
**Estimated Time:** 3 days  
**Dependencies:** Tasks 4.3.1-4.3.3

- [ ] Enhance `src/backtesting/reports/report_generator.py`:
  - [ ] Add executive summary section
  - [ ] Add Monte Carlo confidence intervals
  - [ ] Add statistical significance tests
  - [ ] Add factor exposure analysis
  - [ ] Add trade analysis breakdown
  - [ ] Add risk decomposition
  - [ ] Add benchmark comparison
- [ ] Create PDF export capability
- [ ] Add interactive HTML with Plotly
- [ ] Create email-ready summary format
- [ ] Add scheduling for automated reports

### Task 4.4.2: Build Performance Dashboard
**Priority:** 🟡 Medium  
**Estimated Time:** 3 days  
**Dependencies:** Task 4.4.1

- [ ] Create Streamlit/Dash dashboard:
  - [ ] Real-time equity curve
  - [ ] Drawdown visualization
  - [ ] Position heat map
  - [ ] Trade scatter plot
  - [ ] Rolling metrics charts
  - [ ] Factor exposure over time
  - [ ] Monte Carlo cone
- [ ] Add strategy comparison view
- [ ] Create drill-down capabilities
- [ ] Add export functionality

---

# FINAL CHECKLIST

## Pre-Launch Verification

- [ ] All unit tests passing (>90% coverage)
- [ ] All integration tests passing
- [ ] Performance benchmarks documented
- [ ] API documentation complete
- [ ] Architecture documentation updated
- [ ] Configuration guide created
- [ ] Deployment guide written
- [ ] Disaster recovery plan documented

## Code Quality

- [ ] No critical security vulnerabilities
- [ ] All type hints complete
- [ ] No circular imports
- [ ] Memory leak testing passed
- [ ] Load testing completed
- [ ] Error handling comprehensive

## Documentation

- [ ] README updated with new features
- [ ] API reference generated
- [ ] User guide written
- [ ] Developer guide written
- [ ] Mathematical methodology documented
- [ ] Configuration options documented

---

## Quick Reference: Key Files to Create

| Phase | File Path | Purpose |
|-------|-----------|---------|
| 1 | `src/data/storage/timescale_client.py` | TimescaleDB interface |
| 1 | `src/data/loaders/polars_loader.py` | Polars data loading |
| 1 | `src/training/experiment_tracker.py` | MLflow integration |
| 2 | `src/training/trainer.py` | Main training orchestrator |
| 2 | `src/training/model_factory.py` | Model creation factory |
| 2 | `src/training/validation.py` | Purged CV strategies |
| 2 | `src/training/optimization.py` | Optuna HPO |
| 2 | `src/training/deep_learning/lstm.py` | LSTM model |
| 2 | `src/training/deep_learning/transformer.py` | TFT model |
| 3 | `src/features/fractional_diff.py` | Fractional differentiation |
| 3 | `src/features/orderbook_features.py` | Order book features |
| 3 | `src/features/cointegration.py` | Stat-arb features |
| 3 | `src/data/macro/fred_client.py` | FRED data client |
| 4 | `src/backtesting/event_engine.py` | Event-driven backtest |
| 4 | `src/backtesting/market_impact.py` | Almgren-Chriss model |
| 4 | `src/backtesting/monte_carlo.py` | Monte Carlo analysis |

---

**Document Version:** 1.0  
**Last Updated:** December 2025  
**Total Tasks:** 150+  
**Estimated LOC:** 15,000-20,000
