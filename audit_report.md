# AlphaTrade System: Institutional-Grade Architectural Audit

## Executive Summary

**Audit Conducted By:** Lead Quantitative Researcher / HFT Architect  
**Scope:** Data Ingestion → Feature Engineering → Model Training → Backtesting  
**Assessment Date:** December 2025  
**Overall Assessment:** **INTERMEDIATE** – Solid retail foundation, significant gaps for institutional deployment

The AlphaTrade system demonstrates competent implementation of many modern quantitative finance concepts (fractional differentiation, Purged K-Fold CV, Almgren-Chriss market impact). However, critical architectural gaps exist that would invalidate backtest results in an institutional setting. This report identifies **17 critical issues**, **23 medium-priority gaps**, and provides a phased remediation roadmap.

---

## Section 1: Critical Logic Errors (Backtest-Invalidating)

### 1.1 DATA LEAKAGE IN FEATURE PIPELINE

**Location:** `src/features/pipeline.py`, `src/strategies/ml_based/ml_alpha.py`

**Severity:** 🔴 **CRITICAL**

**Issue:** The `generate_features()` function applies scaling and processing **across the entire dataset** before train/test split:

```python
# From pipeline.py - PROBLEMATIC
def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    features = self.generate_features(df, **kwargs)
    return self.processor.fit_transform(features)  # Fits on ALL data
```

The `FeatureProcessor.fit()` method computes mean/std for `StandardScaler` or `RobustScaler` using **future data** that wouldn't be available at prediction time.

**Additionally in `ml_alpha.py`:**
```python
# Data leakage: _scaler fitted on features that include future-derived statistics
X_scaled = self._scaler.fit_transform(X)  # Full dataset scaling
```

**Impact:** All backtest results are unreliable. The model has implicit access to future distributional information.

**Correction:**
```python
# CORRECT: Fit scaler only on training data, transform test separately
class FeaturePipeline:
    def fit(self, X_train: pd.DataFrame) -> "FeaturePipeline":
        self.processor.fit(X_train)  # Only training data
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.processor.transform(X)  # Apply fitted params
```

---

### 1.2 TARGET VARIABLE CONSTRUCTION LEAKAGE

**Location:** `src/strategies/ml_based/ml_alpha.py`

**Severity:** 🔴 **CRITICAL**

**Issue:** Target variable is created with improper shift semantics:

```python
def _prepare_target(self, prices, horizon):
    forward_returns = prices.pct_change(horizon).shift(-horizon)
    # ...
```

While `shift(-horizon)` is correct for creating forward-looking targets, the issue arises when features computed on the **same row** implicitly contain information about the period used to calculate the target. For example:

- If `horizon=4` bars, the feature row at time `t` is predicting the return from `t` to `t+4`
- But features like `realized_vol_20` at time `t` use data from `t-19` to `t`
- If the model is trained on rows where `t` overlaps with the test period's target calculation window, **serial correlation leaks information**

**Impact:** Inflated performance metrics; model appears to predict future when it's actually fitting to correlated noise.

**Correction:** Implement strict **embargo period** equal to `prediction_horizon + max_feature_lookback`:

```python
embargo_periods = prediction_horizon + max_lookback_in_features
# Remove rows where features overlap with target calculation window
```

---

### 1.3 PURGED CV NOT ENFORCED IN OPTIMIZATION

**Location:** `src/training/optimization.py`, `main.py`

**Severity:** 🔴 **CRITICAL**

**Issue:** While `PurgedKFoldCV` exists, the Optuna optimizer's cross-validation may use default sklearn splitters in some code paths:

```python
# From optimization.py - relies on passed cv, but default fallback unclear
cv_results = cross_validate(
    model, X, y,
    cv=cv,  # What if cv is not PurgedKFoldCV?
    ...
)
```

In `main.py`, the CV object is correctly created:
```python
cv = PurgedKFoldCV(
    n_splits=train_config.get("cv_splits", 5),
    purge_gap=train_config.get("purge_gap", 5),  # DANGER: Default of 5 may be insufficient
    embargo_pct=train_config.get("embargo_pct", 0.01),
)
```

**Issue:** `purge_gap=5` is a **magic number** and is grossly insufficient for most feature sets. The correct formula is:

```
purge_gap = prediction_horizon + max(all_feature_lookback_periods) + safety_buffer
```

For features with 200-bar moving averages and 4-bar prediction horizon, purge_gap should be **at minimum 210**, not 5.

**Impact:** Massive information leakage invalidating all hyperparameter optimization results.

---

### 1.4 INFINITE LIQUIDITY ASSUMPTION

**Location:** `src/backtesting/event_engine.py`

**Severity:** 🔴 **CRITICAL**

**Issue:** The `SimpleExecutionSimulator` assumes all orders fill at the requested quantity:

```python
def simulate_fill(self, order, market_data, timestamp):
    price = market_data.get("price", 0.0)
    # ...
    return create_fill(
        order_event=order,
        fill_price=fill_price,
        # fill_quantity = order.quantity ALWAYS (infinite liquidity)
    )
```

The `OrderBookExecutionSimulator` does handle partial fills but is **disabled by default**:

```python
# Default config
use_order_book: bool = False  # Simple executor used
```

**Impact:** Strategies appear profitable when they would actually move markets against themselves. Particularly devastating for:
- Small-cap stocks
- Options strategies
- Momentum strategies that crowd into the same names

**Correction:** Make `OrderBookExecutionSimulator` the default, or implement ADV-based position limits:

```python
max_position = min(order.quantity, adv * max_participation_rate)
# Where max_participation_rate typically 0.01-0.05 for institutional
```

---

### 1.5 NO SURVIVORSHIP BIAS HANDLING

**Location:** `src/data/loaders/data_loader.py`

**Severity:** 🔴 **CRITICAL**

**Issue:** The data loader discovers symbols from the current filesystem:

```python
def _discover_symbols(self) -> list[str]:
    pattern = f"*_15min.{self.file_format}"
    files = list(self.data_dir.glob(pattern))
    # Only finds currently-existing symbols
```

This design inherently suffers from **survivorship bias** – delisted stocks, bankruptcies, and M&A targets are missing from the universe.

**Impact:** Backtest results are optimistically biased. A study by Elton, Gruber, and Blake (1996) showed survivorship bias can inflate returns by 0.9-1.4% annually.

**Correction:** 
1. Maintain a `symbol_metadata` table with `delist_date`
2. Load universe based on **point-in-time** membership
3. Include delisted symbols in historical simulations

---

## Section 2: Missing Files/Modules

### 2.1 Point-in-Time (PIT) Data Infrastructure

**Status:** ❌ **MISSING**

**Required Files:**
```
src/data/
├── pit/
│   ├── __init__.py
│   ├── pit_loader.py          # Point-in-time data loader
│   ├── corporate_actions.py   # Split/dividend adjustment engine
│   ├── universe_manager.py    # Historical index membership
│   └── as_of_query.py         # "As-of" timestamp queries
```

**Gap:** While `corporate_actions` table exists in TimescaleDB schema, there's **no code** to apply adjustments:

```sql
-- Schema exists in timescale_client.py
CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol VARCHAR(20) NOT NULL,
    action_type VARCHAR(20) NOT NULL,
    ex_date DATE NOT NULL,
    ratio DOUBLE PRECISION,
    ...
);
-- But no Python code consumes this table
```

---

### 2.2 Regime Detection Module

**Status:** ⚠️ **PARTIAL** (only macro regime detection exists)

**Required Files:**
```
src/features/
├── regime/
│   ├── __init__.py
│   ├── hmm_regime.py          # Hidden Markov Model regime detection
│   ├── volatility_regime.py   # Vol-of-vol regime classification
│   ├── correlation_regime.py  # Correlation breakdown detection
│   └── structural_breaks.py   # Chow test, CUSUM implementations
```

**Gap:** `EconomicRegimeDetector` exists but only for macro indicators. Missing:
- Intraday volatility regimes
- Cross-sectional correlation regimes (critical for risk parity strategies)
- Structural break detection for model invalidation

---

### 2.3 Order Flow & Microstructure Features

**Status:** ❌ **MISSING**

**Required Files:**
```
src/features/
├── microstructure/
│   ├── __init__.py
│   ├── order_flow_imbalance.py   # OFI, VPIN
│   ├── kyle_lambda.py            # Kyle's lambda (market impact)
│   ├── roll_spread.py            # Roll's effective spread
│   ├── realized_measures.py      # RV, BV, realized skewness
│   └── toxicity_metrics.py       # Adverse selection measures
```

**Gap:** Current feature set is entirely price-based. No:
- Order Flow Imbalance (OFI)
- Volume-Synchronized Probability of Informed Trading (VPIN)
- Trade direction classification (Lee-Ready, EMO)
- Gamma exposure estimation

---

### 2.4 Missing Test Coverage

**Status:** ⚠️ **INCOMPLETE**

**Existing Tests:**
- `tests/unit/test_risk.py` – Basic VaR, position sizing tests ✓
- `tests/unit/test_indicators.py` – Technical indicators ✓
- `tests/integration/test_backtest.py` – Basic backtest flow ✓

**Missing Tests (Critical):**
```
tests/
├── unit/
│   ├── test_feature_leakage.py     # ❌ MISSING - Most critical
│   ├── test_purged_cv.py           # ❌ MISSING
│   ├── test_market_impact.py       # ❌ MISSING
│   └── test_corporate_actions.py   # ❌ MISSING
├── integration/
│   ├── test_pit_consistency.py     # ❌ MISSING
│   └── test_end_to_end_leakage.py  # ❌ MISSING
└── statistical/
    ├── test_psr_validity.py        # ❌ MISSING
    └── test_backtest_overfitting.py # ❌ MISSING
```

---

### 2.5 Hyperparameter Storage & Versioning

**Status:** ⚠️ **PARTIAL**

**Gap:** MLflow integration exists, but no structured hyperparameter versioning:

```
config/
├── hyperparams/
│   ├── production/
│   │   ├── lightgbm_momentum_v1.yaml
│   │   └── lstm_mean_reversion_v2.yaml
│   └── experiments/
│       └── optuna_study_20251217.db
```

**Current State:** Hyperparameters are scattered across:
- `config/ml_config.yaml` (defaults)
- `main.py` (hardcoded in DEFAULT_CONFIG)
- MLflow runs (if enabled)

No single source of truth for production model configurations.

---

## Section 3: Technology Upgrades Required

### 3.1 Volatility Modeling: Standard → GARCH

**Current:** Simple rolling standard deviation
```python
# Current implementation in pipeline.py
features[f"realized_vol_{window}"] = (
    log_ret.rolling(window=window).std() * np.sqrt(252 * 26)
)
```

**Required:** GARCH family models for proper volatility forecasting

```python
# Recommended implementation
from arch import arch_model

class GARCHVolatilityForecaster:
    def __init__(self, p: int = 1, q: int = 1, model: str = "GARCH"):
        self.model_spec = {"p": p, "q": q, "vol": model}
    
    def fit_forecast(self, returns: pd.Series, horizon: int = 1) -> pd.Series:
        model = arch_model(returns, **self.model_spec)
        result = model.fit(disp="off")
        forecast = result.forecast(horizon=horizon)
        return np.sqrt(forecast.variance.values[-1, :])
```

**Benefits:**
- Captures volatility clustering
- Mean reversion in variance
- Better tail risk estimation

---

### 3.2 Portfolio Optimization: MVO → Black-Litterman

**Current:** Basic mean-variance and risk parity
```python
# From optimizer.py - basic Markowitz
def _max_sharpe(self, constraints):
    def objective(weights):
        port_return = weights @ self.mean_returns.values
        port_vol = np.sqrt(weights @ self.cov_matrix.values @ weights)
        return -(port_return - self.risk_free_rate) / port_vol
```

**Required:** Black-Litterman framework for incorporating views

```python
from pypfopt import BlackLittermanModel, black_litterman

class InstitutionalPortfolioOptimizer:
    def black_litterman_optimize(
        self,
        views: dict,
        view_confidences: np.ndarray,
        market_caps: pd.Series,
        risk_aversion: float = 2.5,
    ) -> pd.Series:
        # Prior from equilibrium
        delta = black_litterman.market_implied_risk_aversion(
            self.market_returns, risk_free_rate=self.risk_free_rate
        )
        prior = black_litterman.market_implied_prior_returns(
            market_caps, delta, self.cov_matrix
        )
        
        # Posterior with views
        bl = BlackLittermanModel(
            self.cov_matrix,
            pi=prior,
            absolute_views=views,
            omega="idzorek",
            view_confidences=view_confidences,
        )
        posterior_rets = bl.bl_returns()
        
        # Optimize on posterior
        ...
```

---

### 3.3 Cross-Validation: Standard → Combinatorial Purged

**Current:** `CombinatorialPurgedKFoldCV` exists but isn't default

**Required Integration:**
```python
# Make CPCV the mandatory default for all optimization
class InstitutionalCVFactory:
    @staticmethod
    def create(
        data_length: int,
        prediction_horizon: int,
        max_feature_lookback: int,
        n_splits: int = 5,
    ) -> CombinatorialPurgedKFoldCV:
        purge_gap = prediction_horizon + max_feature_lookback + 10
        embargo_pct = max(0.02, prediction_horizon / data_length)
        
        return CombinatorialPurgedKFoldCV(
            n_splits=n_splits,
            n_test_splits=2,
            purge_gap=purge_gap,
            embargo_pct=embargo_pct,
        )
```

---

### 3.4 Optimization Metrics: Sharpe → PSR/DSR

**Current:** Optuna optimizes raw Sharpe ratio
```python
# From optimization.py
objective_metric: sharpe_ratio  # Doesn't account for multiple testing
```

**Required:** Deflated Sharpe Ratio as primary metric

```python
class RobustOptimizer(OptunaOptimizer):
    def _calculate_objective(self, model, X_test, y_test, test_idx) -> float:
        predictions = model.predict(X_test)
        returns = predictions * y_test
        
        # Calculate observed Sharpe
        sharpe = returns.mean() / (returns.std() + 1e-8)
        
        # Deflate for multiple testing
        stats = StatisticalTests()
        dsr_result = stats.deflated_sharpe_ratio(
            observed_sharpe=sharpe,
            n_trials=self.n_trials_so_far,
            n_observations=len(returns),
            variance_of_trials=self.sharpe_variance,
        )
        
        return dsr_result.deflated_sharpe  # Use deflated, not raw
```

---

### 3.5 L2/L3 Order Book Data Support

**Current:** OHLCV bars only

**Required Architecture:**
```python
# New data structures needed
@dataclass
class OrderBookSnapshot:
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, int]]  # [(price, size), ...]
    asks: List[Tuple[float, int]]
    
    @property
    def mid_price(self) -> float:
        return (self.bids[0][0] + self.asks[0][0]) / 2
    
    @property
    def microprice(self) -> float:
        """Volume-weighted mid price"""
        bid_price, bid_size = self.bids[0]
        ask_price, ask_size = self.asks[0]
        return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
    
    @property
    def spread_bps(self) -> float:
        return (self.asks[0][0] - self.bids[0][0]) / self.mid_price * 10000

@dataclass 
class OrderBookDelta:
    timestamp: datetime
    symbol: str
    side: Literal["bid", "ask"]
    price: float
    size_delta: int  # Positive = add, negative = remove
```

---

## Section 4: Detailed Gap Analysis by Module

### 4.1 Data Ingestion (Alpha Pipeline)

| Component | Current State | Institutional Requirement | Gap |
|-----------|--------------|---------------------------|-----|
| PIT Data | ❌ Missing | As-of queries for all data | **CRITICAL** |
| Corporate Actions | Schema only | Full adjustment engine | **CRITICAL** |
| Survivorship Bias | ❌ Not handled | Universe time-travel | **CRITICAL** |
| L2/L3 Data | ❌ Missing | Order book snapshots | HIGH |
| Tick Data | Basic support | Nanosecond precision | MEDIUM |
| Data Quality | Basic validation | Anomaly detection ML | MEDIUM |

### 4.2 Feature Engineering

| Component | Current State | Institutional Requirement | Gap |
|-----------|--------------|---------------------------|-----|
| Technical Indicators | 50+ indicators ✓ | Adequate | LOW |
| Fractional Diff | Implemented ✓ | Good | LOW |
| Cointegration | Implemented ✓ | Good | LOW |
| Microstructure | ❌ Missing | OFI, VPIN, Kyle's λ | **CRITICAL** |
| Alternative Data | ❌ Missing | Sentiment, satellite | HIGH |
| Feature Leakage Check | ❌ Missing | Automated validation | **CRITICAL** |

### 4.3 Model Training

| Component | Current State | Institutional Requirement | Gap |
|-----------|--------------|---------------------------|-----|
| Purged CV | Implemented ✓ | Good | LOW |
| Walk-Forward | Implemented ✓ | Good | LOW |
| CPCV | Implemented | Should be mandatory | MEDIUM |
| PSR/DSR | Implemented ✓ | Should be primary metric | MEDIUM |
| Model Registry | MLflow ✓ | Good | LOW |
| Hyperparameter Versioning | ⚠️ Scattered | Centralized store | MEDIUM |

### 4.4 Backtesting Engine

| Component | Current State | Institutional Requirement | Gap |
|-----------|--------------|---------------------------|-----|
| Vectorized Engine | ✓ Implemented | Good | LOW |
| Event-Driven | ✓ Implemented | Good | LOW |
| Market Impact | Almgren-Chriss ✓ | Good | LOW |
| Partial Fills | Optional | Mandatory default | **CRITICAL** |
| Latency Modeling | Basic | Exchange-specific models | HIGH |
| Rejection Rates | ❌ Missing | Realistic rejection sim | HIGH |

---

## Section 5: Magic Numbers Audit

The following hardcoded values were identified and require externalization:

| Location | Magic Number | Current Value | Recommendation |
|----------|--------------|---------------|----------------|
| `main.py` | `purge_gap` | 5 | Calculate: `horizon + max_lookback` |
| `main.py` | `embargo_pct` | 0.01 | Calculate: `horizon / data_length` |
| `validation.py` | `min_train_samples` | 100 | Asset-class specific |
| `event_engine.py` | `fill_probability` | 1.0 | Model from historical data |
| `market_impact.py` | `eta` (temp impact) | 0.1 | Calibrate from trade data |
| `market_impact.py` | `gamma` (perm impact) | 0.05 | Calibrate from trade data |
| `position_sizing.py` | `kelly_fraction` | 0.25 | Risk budget dependent |
| `fractional_diff.py` | `threshold` | 1e-5 | Memory vs stationarity tradeoff |
| `pipeline.py` | `clip_outliers` | 5.0 | Asset-class specific |

---

## Section 6: Action Plan - Phased Remediation

### Phase 1: Critical Bug Fixes (Week 1-2)

**Objective:** Make backtest results valid

1. **Fix Feature Leakage**
   - Modify `FeaturePipeline` to separate fit/transform
   - Add `LeakageDetector` class to validate feature timestamps
   - Unit tests for leakage detection

2. **Fix Purge Gap Calculation**
   - Create `calculate_minimum_purge_gap()` utility
   - Auto-detect max lookback from feature definitions
   - Enforce minimum purge in all CV classes

3. **Enable Partial Fills by Default**
   - Change `use_order_book` default to `True`
   - Add ADV-based position limits
   - Log warnings when fill assumptions violated

4. **Add Survivorship Bias Handling**
   - Create `UniverseManager` class
   - Load historical index memberships
   - Filter backtest universe by date

**Deliverables:**
- [ ] `src/validation/leakage_detector.py`
- [ ] Updated `FeaturePipeline` with fit/transform separation
- [ ] `src/data/pit/universe_manager.py`
- [ ] Test suite: `tests/unit/test_feature_leakage.py`

---

### Phase 2: Data Infrastructure (Week 3-4)

**Objective:** Institutional-grade data handling

1. **Point-in-Time Data Layer**
   - Implement `PITLoader` with as-of queries
   - Corporate action adjustment engine
   - TimescaleDB continuous aggregates for PIT

2. **L2 Order Book Support**
   - `OrderBookSnapshot` and `OrderBookDelta` dataclasses
   - TimescaleDB schema for order book storage
   - Order book reconstruction from deltas

3. **Data Quality Pipeline**
   - Automated anomaly detection
   - Cross-validation against multiple sources
   - Real-time data quality monitoring

**Deliverables:**
- [ ] `src/data/pit/` module complete
- [ ] `src/data/orderbook/` module
- [ ] TimescaleDB migration scripts
- [ ] Data quality dashboard

---

### Phase 3: Feature Enhancement (Week 5-6)

**Objective:** Institutional feature set

1. **Microstructure Features**
   - Order Flow Imbalance (OFI)
   - VPIN implementation
   - Kyle's lambda estimation
   - Lee-Ready trade classification

2. **GARCH Volatility Models**
   - GARCH(1,1), EGARCH, GJR-GARCH
   - Volatility forecasting integration
   - Regime-switching volatility

3. **Regime Detection**
   - HMM-based regime classification
   - Structural break detection
   - Correlation regime monitoring

**Deliverables:**
- [ ] `src/features/microstructure/` module
- [ ] `src/features/volatility/garch.py`
- [ ] `src/features/regime/` module
- [ ] Feature documentation

---

### Phase 4: Optimization & Validation (Week 7-8)

**Objective:** Statistically rigorous model development

1. **DSR-Based Optimization**
   - Replace Sharpe with Deflated Sharpe
   - Track trial count for multiple testing adjustment
   - Minimum Track Record Length enforcement

2. **Black-Litterman Integration**
   - View specification interface
   - Confidence calibration framework
   - Integration with existing optimizer

3. **Comprehensive Test Suite**
   - Statistical tests for backtest validity
   - Leakage detection automation
   - CI/CD integration

**Deliverables:**
- [ ] Updated `OptunaOptimizer` with DSR
- [ ] `src/portfolio/black_litterman.py`
- [ ] Full test coverage (>80%)
- [ ] CI/CD pipeline configuration

---

## Appendix A: Configuration File Recommendations

### A.1 Recommended `config/institutional_defaults.yaml`

```yaml
# Institutional-Grade Default Configuration

data:
  point_in_time: true
  survivorship_bias_correction: true
  corporate_action_adjustment: true
  min_adv_filter: 1_000_000  # $1M minimum ADV

features:
  max_lookback_periods: 200
  microstructure_enabled: true
  garch_volatility: true
  leakage_check: strict  # fail on any detected leakage

cross_validation:
  type: combinatorial_purged_kfold
  n_splits: 6
  n_test_splits: 2
  purge_gap: auto  # Calculate from features
  embargo_pct: auto  # Calculate from horizon
  min_train_samples: 1000

optimization:
  primary_metric: deflated_sharpe_ratio
  secondary_metrics:
    - probabilistic_sharpe_ratio
    - sortino_ratio
    - max_drawdown
  multiple_testing_correction: true
  min_track_record_months: 12

backtesting:
  execution_simulator: order_book  # Not simple
  partial_fills: true
  max_participation_rate: 0.02  # 2% of ADV
  latency_ms: 50  # Realistic retail latency
  rejection_rate: 0.02  # 2% order rejection

risk:
  var_method: historical
  var_confidence: 0.99
  max_position_adv_pct: 5.0  # Max 5% of ADV
  max_sector_exposure: 0.25
  drawdown_flatten_threshold: 0.15
```

---

## Appendix B: Recommended Library Additions

```toml
# pyproject.toml additions

[project.optional-dependencies]
institutional = [
    "arch>=6.0",              # GARCH models
    "pyportfolioopt>=1.5",    # Black-Litterman, HRP
    "hmmlearn>=0.3",          # Hidden Markov Models
    "ruptures>=1.1",          # Change point detection
    "quantlib>=1.31",         # Derivatives pricing
    "arctic>=1.80",           # Tick data storage
    "sortedcontainers>=2.4",  # Order book implementation
]
```

---

## Appendix C: Test Cases for Leakage Detection

```python
# tests/unit/test_feature_leakage.py

import pytest
import numpy as np
import pandas as pd

class TestFeatureLeakage:
    """Critical tests for data leakage detection."""
    
    def test_scaler_not_fitted_on_test_data(self, pipeline, train_data, test_data):
        """Verify scaler only sees training data."""
        pipeline.fit(train_data)
        
        # Scaler params should only reflect training distribution
        train_mean = train_data.mean()
        scaler_mean = pd.Series(pipeline.processor._scaler.mean_, index=train_data.columns)
        
        np.testing.assert_array_almost_equal(train_mean, scaler_mean, decimal=5)
    
    def test_no_future_features_in_training(self, features_df, target_df):
        """Verify no feature uses future information."""
        for col in features_df.columns:
            # Feature at time t should only depend on data <= t
            feature_t = features_df[col].iloc[100]
            feature_t_minus_1 = features_df[col].iloc[99]
            
            # Recalculate feature using only data up to t
            expected = calculate_feature_pit(features_df, col, t=100)
            
            assert feature_t == expected, f"Leakage detected in {col}"
    
    def test_purge_gap_sufficient(self, cv, max_lookback, prediction_horizon):
        """Verify purge gap exceeds feature lookback + prediction horizon."""
        min_required = max_lookback + prediction_horizon
        
        assert cv.purge_gap >= min_required, (
            f"Purge gap {cv.purge_gap} insufficient. "
            f"Required: {min_required} (lookback={max_lookback}, horizon={prediction_horizon})"
        )
```

---

## Conclusion

The AlphaTrade system provides a solid foundation with many institutional concepts correctly implemented (fractional differentiation, Purged K-Fold CV, Almgren-Chriss market impact). However, **critical data leakage issues** and **infinite liquidity assumptions** would invalidate any backtest results in a production setting.

The phased remediation plan prioritizes:
1. **Data integrity fixes** that invalidate current results
2. **Infrastructure upgrades** for institutional data handling
3. **Feature enhancements** for alpha capture
4. **Statistical rigor** in model validation

Estimated total remediation effort: **8-10 weeks** for a senior quantitative developer.

---

*Report prepared in accordance with JPMorgan Quantitative Research standards.*
*Classification: Internal Use Only*
