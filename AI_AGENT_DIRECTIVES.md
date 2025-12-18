# ALPHATRADE SYSTEM - AI AGENT IMPROVEMENT DIRECTIVES

## Document Purpose
This document contains directives for an AI Agent to improve AlphaTrade System to JPMorgan institutional level. Focus areas: Features and Training modules with their dependencies.

---

# SECTION 1: CRITICAL BUGS TO FIX IMMEDIATELY

## Bug 1.1: Incorrect max_lookback Calculation
**Location**: src/features/pipeline.py, Line ~420
**Problem**: The max_lookback only considers MA periods but ignores other feature lookbacks like Hurst (100 bars), autocorrelation (50 bars), and lag features.
**Impact**: Causes data leakage because purge_gap calculation uses this incorrect value.
**Directive**: Modify max_lookback calculation to include ALL rolling window sizes, shift operations, and lag periods used anywhere in feature generation. Add 50 bars safety buffer.

## Bug 1.2: Purge Gap Auto-Calculation Too Small
**Location**: src/training/training_pipeline.py, Line ~590
**Problem**: When purge_gap is set to "auto", it calculates as prediction_horizon * 2 + 5, which completely ignores feature lookback.
**Impact**: Severe data leakage. A 5-bar prediction with 200-bar MA gets only ~15 bar purge instead of required ~215.
**Directive**: Change auto calculation formula to: purge_gap = prediction_horizon + max_feature_lookback + buffer. Coordinate with feature pipeline to get actual max_lookback value.

## Bug 1.3: Feature Pipeline Allows Direct Leakage
**Location**: src/features/pipeline.py
**Problem**: generate_features() method is public and can be called on entire dataset before train/test split, causing scaling parameter leakage.
**Impact**: Model sees future statistical information during training.
**Directive**: Make generate_features() private. Only expose fit(), transform(), and fit_transform() methods. Add runtime checks to prevent transform() before fit().

## Bug 1.4: No Multi-Asset Timestamp Alignment Validation
**Location**: src/training/training_pipeline.py
**Problem**: When training on multiple symbols, no validation ensures timestamps are aligned and no cross-asset future data contamination.
**Impact**: One asset's future data could leak into another asset's features.
**Directive**: Add timestamp alignment validation in _prepare_training_data() method. Verify all symbols have identical timestamp ranges. Add cross-asset leakage detection.

---

# SECTION 2: FEATURES MODULE IMPROVEMENTS

## Directive 2.1: Expand Time Cyclical Encoder
**Location**: src/features/transformers.py
**Current State**: Only encodes hour and day_of_week.
**Required Additions**:
- Minute of hour encoding (sin/cos)
- Market session indicator (pre-market=0, regular=1, after-hours=2)
- Time to market open (minutes, negative after open)
- Time to market close (minutes)
- First 30 minutes of session flag
- Last 30 minutes of session flag
- Quarter of year encoding (sin/cos)
- Options expiration week flag
- FOMC announcement week flag

## Directive 2.2: Add Adaptive Technical Indicators
**Location**: src/features/technical/indicators.py
**Required Additions**:
- Kaufman Adaptive Moving Average (KAMA) - adapts to market noise
- Fractal Adaptive Moving Average (FRAMA) - uses fractal dimension
- Variable Index Dynamic Average (VIDYA) - volatility-adjusted
- McGinley Dynamic indicator - smoother than EMA
- Ehlers filters (instantaneous trendline, cyber cycle)

## Directive 2.3: Add Multi-Timeframe Features
**Location**: Create new file src/features/technical/multi_timeframe.py
**Required Features**:
- RSI from multiple timeframes (5min, 15min, 1h, 4h, 1d) aligned to base timeframe
- Trend alignment score (+1 if all timeframes bullish, -1 if all bearish, 0 mixed)
- Higher timeframe support/resistance levels
- Cross-timeframe momentum divergence detection

## Directive 2.4: Expand Microstructure Features
**Location**: src/features/microstructure/
**Required Additions**:
- Order Book Imbalance at multiple depth levels (1, 3, 5, 10)
- Weighted depth imbalance across all visible levels
- Bid-ask spread percentile (vs recent history)
- Spread momentum (change in spread)
- Trade flow toxicity (beyond VPIN)
- Smart money indicator (large trades at bid vs ask)
- Institutional activity index (block trade detection)

## Directive 2.5: Integrate Fractional Differentiation
**Location**: src/features/pipeline.py
**Current State**: fractional_diff.py exists but is NOT integrated into main pipeline.
**Directive**: Add fractional differentiation as optional step in FeaturePipeline. Apply to price columns (open, high, low, close) and volume. Use auto_find_d for optimal differentiation order. Add configuration option in ml_config.yaml to enable/disable.

## Directive 2.6: Add SHAP-Based Feature Selection
**Location**: src/features/feature_selection.py
**Current State**: Uses basic importance methods.
**Required Additions**:
- SHAP value calculation for feature selection
- Temporal feature importance tracking (detect decaying features)
- Automatic removal of features losing predictive power
- Adversarial feature validation (detect distribution shift features)

## Directive 2.7: Create Market Regime Detection
**Location**: Create new file src/features/regime/regime_detector.py
**Required Components**:
- Hidden Markov Model for regime classification
- Three regimes: low volatility trending, normal, high volatility/crisis
- Regime transition probability matrix
- Regime-conditional feature generation
- Integration with main feature pipeline

---

# SECTION 3: TRAINING MODULE IMPROVEMENTS

## Directive 3.1: Add GroupTimeSeriesSplit
**Location**: src/training/validation.py
**Current State**: Has PurgedKFoldCV and WalkForward but no multi-asset group handling.
**Required**: Create GroupTimeSeriesSplit class that ensures no asset appears in both train and test within same fold. Critical for panel data with multiple symbols.

## Directive 3.2: Make Combinatorial CV Default
**Location**: src/training/training_pipeline.py
**Current State**: CombinatorialPurgedKFoldCV exists but standard PurgedKFoldCV is default.
**Directive**: Change default cv_type to "combinatorial_purged" in training configuration. This provides multiple backtest paths for robust performance estimation.

## Directive 3.3: Add Adversarial Validation Stage
**Location**: src/training/training_pipeline.py
**Required**: Add new pipeline stage that trains classifier to distinguish train from test data. If AUC exceeds 0.55 log warning, if exceeds 0.60 log critical error. Use RandomForest for fast detection.

## Directive 3.4: Add Drift Detection Stage
**Location**: src/training/training_pipeline.py
**Required**: Add stage using Population Stability Index (PSI) and Kolmogorov-Smirnov test to detect concept drift between train and test periods. Flag features with significant drift.

## Directive 3.5: Add Meta-Labeling Support
**Location**: src/training/training_pipeline.py
**Required**: Implement meta-labeling from de Prado. Train secondary model to predict if primary model's prediction will be correct. Use for position sizing - bet size proportional to confidence.

## Directive 3.6: Add Triple-Barrier Labeling
**Location**: Create new file src/training/target_engineering.py
**Required**: Implement triple-barrier labeling method. Labels based on which barrier hit first: upper (take profit) = +1, lower (stop loss) = -1, time barrier = 0. Better than simple direction labels.

## Directive 3.7: Add Neural Network Models to Factory
**Location**: src/training/model_factory.py
**Required Additions to ModelType enum**:
- LSTM
- GRU  
- Transformer
- Temporal Fusion Transformer
- WaveNet
- N-BEATS

## Directive 3.8: Add Ensemble Factory Methods
**Location**: src/training/model_factory.py
**Required Methods**:
- create_stacking_ensemble with configurable meta-learner
- create_blending_ensemble with optimizable weights
- create_boosted_ensemble (sequential boosting of different model types)

## Directive 3.9: Add Statistical Significance Tests
**Location**: src/training/training_pipeline.py
**Required Tests**:
- Probability of Backtest Overfitting (PBO) calculation
- Deflated Sharpe Ratio (adjust for multiple testing)
- Minimum Track Record Length calculation
- Combinatorial Symmetric Cross-Validation test

## Directive 3.10: Add Model Explanation Module
**Location**: Create new file src/training/model_explanation.py
**Required Components**:
- SHAP TreeExplainer integration
- Feature importance visualization
- Prediction explanation for individual samples
- Full explanation report generation (HTML/PDF)

---

# SECTION 4: MAIN.PY ISSUES

## Issue 4.1: Data Validation Insufficient
**Location**: main.py, load_data function
**Problem**: Validation is basic, missing critical checks for trading data.
**Required Additions**:
- Check for stock splits (price jumps > 20% with volume spike)
- Check for dividend adjustments
- Verify no future timestamps
- Check for weekend/holiday data that should not exist
- Validate OHLC relationships (high >= low, high >= open/close)

## Issue 4.2: Feature Generation Split Ratio Hardcoded
**Location**: main.py, generate_features function
**Problem**: train_ratio=0.8 is hardcoded, should match actual training split.
**Directive**: Pass actual train/test split configuration to generate_features. Ensure feature pipeline fit uses exact same data that will be used for model training.

## Issue 4.3: No Feature Importance Analysis After Training
**Location**: main.py, train_ml_model function
**Problem**: Model trains but feature importance is not analyzed or logged.
**Directive**: Add feature importance extraction after training. Log top 20 features. Save importance plot. Alert if top features are unexpected (like pure price columns).

## Issue 4.4: Missing Model Persistence Validation
**Location**: main.py
**Problem**: Models saved without validation that they can be loaded correctly.
**Directive**: After saving model, immediately load it back and verify predictions match. Log model size and loading time.

## Issue 4.5: No Production Readiness Checks
**Location**: main.py
**Problem**: No checks before deploying trained model.
**Required Checks**:
- Model file exists and loads
- Feature pipeline can transform new data
- Prediction latency within acceptable bounds
- Model produces valid output range
- All required features available in production data

---

# SECTION 5: CONFIGURATION IMPROVEMENTS

## Directive 5.1: Add Regime-Specific Parameters
**Location**: config/ml_config.yaml
**Required**:
- Market regime detection configuration
- Per-regime position scaling
- Per-regime prediction horizon
- Per-regime model parameters

## Directive 5.2: Add Online Learning Configuration
**Location**: config/ml_config.yaml
**Required**:
- Update frequency (daily, weekly, on_drift)
- Minimum samples for update
- Maximum model age before forced retrain
- Drift detection parameters (ADWIN, DDM thresholds)
- Incremental training settings

## Directive 5.3: Add Feature Store Configuration
**Location**: config/ml_config.yaml
**Required**:
- Backend selection (TimescaleDB, Redis, Parquet)
- Cache settings (TTL, max size)
- Feature versioning settings
- Monitoring and alerting for stale/missing features

## Directive 5.4: Add Monitoring Configuration
**Location**: config/ml_config.yaml
**Required**:
- Model performance monitoring thresholds
- Feature drift alert thresholds
- Prediction latency thresholds
- Alert notification settings (email, Slack, PagerDuty)

---

# SECTION 6: MISSING CRITICAL MODULES

## Module 6.1: Real-Time Feature Pipeline
**Location**: Create src/features/realtime_pipeline.py
**Purpose**: Feature computation optimized for real-time prediction with O(1) incremental updates.
**Requirements**:
- Incremental rolling calculations (no history recalculation)
- Memory-efficient state management
- Latency monitoring (alert if exceeds threshold)
- Exact same output as batch pipeline (consistency verification)

## Module 6.2: Model Monitoring
**Location**: Create src/training/model_monitoring.py
**Purpose**: Monitor deployed model performance in production.
**Requirements**:
- Log every prediction with features
- Rolling performance calculation
- Feature drift detection using PSI
- Performance degradation alerts
- Daily monitoring report generation

## Module 6.3: Position Sizing Integration
**Location**: Create src/training/ml_position_sizing.py
**Purpose**: Integrate ML confidence with position sizing.
**Requirements**:
- Meta-model training for correctness prediction
- Kelly fraction calculation
- Confidence-weighted position sizes
- Risk-adjusted sizing based on regime

---

# SECTION 7: IMPLEMENTATION PRIORITIES

## Priority 1 - CRITICAL (Complete within 48 hours)
1. Fix max_lookback calculation in feature pipeline
2. Fix purge_gap auto-calculation in training pipeline
3. Add leakage prevention enforcement
4. Add multi-asset alignment validation

## Priority 2 - HIGH (Complete within 1 week)
5. Integrate fractional differentiation
6. Add adversarial validation stage
7. Expand time features
8. Add SHAP-based feature selection and explanation
9. Make combinatorial CV default

## Priority 3 - MEDIUM (Complete within 2 weeks)
10. Create market regime detection module
11. Add meta-labeling support
12. Add neural network models to factory
13. Create real-time feature pipeline
14. Add model monitoring module

## Priority 4 - LOW (Backlog)
15. Add symbolic regression for feature discovery
16. Expand LOB microstructure features
17. Add online learning capability
18. Create full production deployment pipeline

---

# SECTION 8: CODE QUALITY REQUIREMENTS

All modifications must meet these standards:
- 100% type hint coverage with mypy verification
- Numpy-style docstrings on all public methods
- Unit tests with >80% coverage
- Structured logging with correlation IDs
- No silent failures - explicit exception handling
- Profiled hot paths with <100ms prediction latency
- No hardcoded credentials
- Deterministic results with fixed random seeds
- All models have drift detection and alerting

---

# SECTION 9: TESTING REQUIREMENTS

## Required Test Coverage
- Unit tests for all new feature calculations
- Integration tests for full pipeline (data → features → model → prediction)
- Leakage detection tests (verify no future data in features)
- Performance tests (verify latency requirements)
- Regression tests (verify existing functionality not broken)

## Specific Test Cases Required
- Test purge_gap correctly prevents leakage
- Test max_lookback includes all feature lookbacks
- Test feature pipeline fit/transform separation
- Test multi-asset timestamp alignment
- Test adversarial validation detects distribution shift
- Test meta-labeling improves position sizing
- Test regime detection produces stable regimes

---

**Document Version**: 1.0
**Created**: December 2024
**Target System**: AlphaTrade v2.0
**Goal**: JPMorgan Institutional Level Quality
