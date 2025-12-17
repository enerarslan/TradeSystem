📋 GENERAL INSTRUCTIONS
This task plan contains all necessary fixes and improvements to transform the AlphaTrade system into a JPMorgan-level institutional trading platform. Execute each step sequentially, preserve existing architecture when writing code, and verify imports/dependencies after each change.
Project Directory: C:\Users\enera\Desktop\AlphaTrade_System
Training Data Context: 4.5 years of 15-minute OHLCV data. Market hours timing for training execution is not a priority at this stage.
Language: Write all code, comments, and documentation in English.

🔴 SECTION 1: CRITICAL BUG FIXES (DO THESE FIRST)
1.1 Feature Pipeline Indentation Error
File: src/features/pipeline.py
Problem: The generate_features method is defined OUTSIDE the FeaturePipeline class due to incorrect indentation. Around line 230, def generate_features( is not properly indented inside the class.
Tasks:

Locate the generate_features method (approximately line 230)
Move it INSIDE the FeaturePipeline class with proper indentation (same level as fit, transform, fit_transform methods)
Ensure self parameter is first parameter
Verify the method is callable via pipeline.generate_features(df)
Test that fit(), transform(), and fit_transform() methods can call generate_features() internally
Check all internal method calls within the class use self.generate_features()


1.2 Microstructure Module Export Error
File: src/features/microstructure/__init__.py
Problem: OrderBookDynamics is imported but NOT included in the __all__ list, making it unavailable for external imports.
Tasks:

Add "OrderBookDynamics" to the __all__ list
Verify the import statement from .order_book_dynamics import OrderBookDynamics exists
Test import works: from src.features.microstructure import OrderBookDynamics


1.3 ModelFactory Wrong Method Call
File: main.py
Problem: Code calls ModelFactory.create() but the correct method name is ModelFactory.create_model().
Tasks:

Search entire file for ModelFactory.create( (without "model")
Replace all occurrences with ModelFactory.create_model(
Fix parameter passing: change ModelFactory.create(model_type, **best_params) to ModelFactory.create_model(model_type, params=best_params)
This error appears around lines 760, 820, and potentially elsewhere
Verify all calls match the method signature in src/training/model_factory.py


1.4 Configuration File Mismatch
Files:

config/ml_config.yaml
main.py (DEFAULT_CONFIG dictionary)

Problem: Config file has purge_gap: 10 and max_feature_lookback: 20, but main.py uses max_lookback_periods: 200. This mismatch can cause data leakage.
Tasks:

Update config/ml_config.yaml in the cross_validation section:

yaml  cross_validation:
    n_splits: 5
    purge_gap: "auto"              # Changed from 10
    embargo_pct: 0.01
    prediction_horizon: 5          # Changed from 1
    max_feature_lookback: 200      # Changed from 20
```
- Ensure `main.py` DEFAULT_CONFIG matches these values
- Verify the `calculate_purge_gap()` function in `main.py` uses these config values
- Add validation to warn if loaded config has insufficient purge_gap

---

### 1.5 Features Module Export Completeness

**File:** `src/features/__init__.py`

**Problem:** `TimeCyclicalEncoder` is imported but not in `__all__`. Microstructure features are not exported.

**Tasks:**
- Add to `__all__` list:
  - `"TimeCyclicalEncoder"`
  - `"OrderFlowImbalance"`
  - `"calculate_ofi"`
  - `"VPIN"`
  - `"calculate_vpin"`
  - `"KyleLambda"`
  - `"calculate_kyle_lambda"`
  - `"RollSpread"`
  - `"AmihudIlliquidity"`
  - `"OrderBookDynamics"`
- Add corresponding import statements from microstructure module
- Test all imports work correctly

---

### 1.6 Training Result Object Usage

**File:** `main.py` → `train_ml_model()` function

**Problem:** The `TrainingResult` object is created with parameters that may not match the dataclass definition.

**Tasks:**
- Check `src/training/trainer.py` for `TrainingResult` dataclass definition
- Verify all parameters passed to `TrainingResult()` in `main.py` match the dataclass fields
- Ensure `cv_scores` is passed as numpy array, not dict (check expected type)
- Add any missing required fields
- Remove any fields that don't exist in the dataclass

---

### 1.7 Deep Learning Model Save Error

**File:** `main.py` → training mode section (around line 900)

**Problem:** Deep learning model saving uses `torch.save(model, save_path)` but torch may not be imported, and the model object type should be verified.

**Tasks:**
- Add proper torch import at the top of the file (with try/except for optional dependency)
- Verify model type before saving (LSTM, Transformer, etc.)
- Use proper save method based on model type:
  - PyTorch models: `torch.save(model.state_dict(), path)` for weights only
  - Or `torch.save({'model': model, 'config': config}, path)` for full checkpoint
- Add error handling for save operation
- Create models directory if it doesn't exist

---

## 🟠 SECTION 2: TRAINING SYSTEM IMPROVEMENTS

### 2.1 Complete Deep Learning Training Implementation

**File:** `main.py` → `train_deep_learning()` function

**Problem:** Function creates model but does NOT train it. Returns untrained model.

**Tasks:**

1. **Add Complete Training Loop:**
   - Import necessary PyTorch modules (torch, torch.nn, torch.optim)
   - Import DataLoader utilities from `src/training/deep_learning/dataset.py`
   - Create proper train/validation split with purging
   - Implement epoch loop with:
     - Training phase (model.train())
     - Validation phase (model.eval())
     - Loss calculation using financial losses from `src/training/deep_learning/losses.py`
     - Backpropagation and optimizer step
     - Gradient clipping (use config value)
     - Learning rate scheduling

2. **Add Training Features:**
   - Early stopping based on validation loss (patience from config)
   - Best model checkpoint saving
   - Training history logging (loss, metrics per epoch)
   - Memory management (clear cache periodically)
   - Progress logging every N epochs

3. **Use Custom Financial Losses:**
   - Import `SharpeLoss`, `SortinoLoss`, `CombinedFinancialLoss` from deep_learning.losses
   - Allow loss function selection via config
   - Combine MSE with financial loss if needed

4. **Return Proper Result:**
   - Return trained model (not untrained)
   - Include training history
   - Include best validation metrics
   - Include training time

---

### 2.2 Model Drift Detection Module

**New File:** `src/training/drift_detection.py`

**Purpose:** Detect when model performance degrades and retraining is needed.

**Implementation Requirements:**

1. **DriftDetector Class:**
```
   class DriftDetector:
       - __init__(self, reference_data, thresholds)
       - detect_feature_drift(current_data) -> DriftResult
       - detect_prediction_drift(predictions) -> DriftResult
       - detect_performance_drift(metrics) -> DriftResult
       - get_drift_report() -> Dict
```

2. **Drift Metrics to Implement:**
   - Population Stability Index (PSI) for each feature
   - Kolmogorov-Smirnov test for distribution comparison
   - Jensen-Shannon divergence
   - Performance metric tracking (Sharpe ratio degradation, accuracy drop)

3. **DriftResult Dataclass:**
```
   @dataclass
   class DriftResult:
       is_drift_detected: bool
       drift_score: float
       drift_type: str  # "feature", "prediction", "performance"
       affected_features: List[str]
       severity: str  # "low", "medium", "high", "critical"
       recommendation: str  # "monitor", "retrain", "investigate"
```

4. **Threshold Configuration:**
   - PSI > 0.1: Warning
   - PSI > 0.25: Critical
   - Make thresholds configurable

5. **Integration Points:**
   - Can be called before training to check if retraining needed
   - Can be run on schedule to monitor production models
   - Logs results to MLflow/experiment tracker

---

### 2.3 Training Pipeline Orchestrator

**New File:** `src/training/training_pipeline.py`

**Purpose:** Orchestrate complete training workflow with proper error handling.

**Implementation Requirements:**

1. **TrainingPipeline Class:**
```
   class TrainingPipeline:
       - __init__(self, config)
       - run(data, mode="full") -> PipelineResult
       - validate_data(data) -> ValidationResult
       - generate_features(data) -> Tuple[features, pipelines]
       - prepare_training_data(features) -> Tuple[X, y, cv]
       - train_model(X, y, cv) -> TrainingResult
       - evaluate_model(model, X_test, y_test) -> EvaluationResult
       - register_model(model, metrics) -> str  # returns model_id
       - cleanup()
```

2. **Pipeline Steps:**
   - Step 1: Data validation (quality checks, missing data, outliers)
   - Step 2: Feature generation with leakage prevention
   - Step 3: Train/test split with proper purging
   - Step 4: Model training with cross-validation
   - Step 5: Out-of-sample evaluation
   - Step 6: Statistical significance tests
   - Step 7: Model registration if metrics pass threshold
   - Step 8: Cleanup temporary files

3. **Error Handling:**
   - Try/except around each step
   - Detailed error logging
   - Partial result saving on failure
   - Rollback capability

4. **Logging:**
   - Log start/end of each step with timing
   - Log intermediate metrics
   - Log data shapes at each step
   - Log any warnings or anomalies

---

### 2.4 Hyperparameter Optimization Enhancement

**File:** `src/training/optimization.py`

**Tasks:**

1. **Add Walk-Forward Optimization:**
   - Optimize hyperparameters using walk-forward validation (not just k-fold)
   - This prevents overfitting to specific time periods
   - Use same purge gap as training

2. **Add Multi-Metric Optimization:**
   - Optimize for multiple objectives: Sharpe, Sortino, Max Drawdown
   - Use Pareto frontier for multi-objective
   - Allow configurable metric weights

3. **Add Optuna Pruning:**
   - Implement early pruning for bad trials
   - Use MedianPruner or HyperbandPruner
   - Save computation time

4. **Add Hyperparameter Importance:**
   - After optimization, calculate which hyperparameters matter most
   - Log importance to MLflow
   - Help understand model behavior

---

### 2.5 Cross-Validation Improvements

**File:** `src/training/validation.py`

**Tasks:**

1. **Add Purge Gap Validation:**
   - Add method `validate_purge_gap(purge_gap, max_lookback, prediction_horizon)`
   - Raise warning if purge_gap < max_lookback + prediction_horizon
   - Log the calculation details

2. **Add Split Visualization:**
   - Add method to visualize train/test splits
   - Show purge and embargo regions clearly
   - Save visualization as image file

3. **Add Leakage Detection in Splits:**
   - After generating splits, verify no index overlap
   - Verify temporal ordering (train indices < test indices after purge)
   - Raise error if leakage detected

4. **Add Time-Based Stratification:**
   - Option to stratify by time periods (ensure each fold has different market regimes)
   - Option to stratify by volatility regime
   - Improve generalization

---

## 🟡 SECTION 3: FEATURE ENGINEERING IMPROVEMENTS

### 3.1 Feature Leakage Prevention System

**New File:** `src/features/leakage_prevention.py`

**Purpose:** Systematically prevent and detect look-ahead bias in features.

**Implementation Requirements:**

1. **LeakageChecker Class:**
```
   class LeakageChecker:
       - check_feature_for_leakage(feature_series, price_series, target_series) -> LeakageResult
       - check_all_features(feature_df, price_df, target_series) -> Dict[str, LeakageResult]
       - get_safe_features(feature_df, ...) -> List[str]
       - generate_report() -> str
```

2. **Leakage Detection Methods:**
   - **Future Correlation Test:** Check if feature correlates with future returns (should not)
   - **Timestamp Validation:** Verify feature timestamps <= current bar timestamp
   - **Rolling Window Validation:** Verify rolling calculations don't include current bar in some cases
   - **Target Leakage Test:** Check if feature contains information about target variable

3. **LeakageResult Dataclass:**
```
   @dataclass
   class LeakageResult:
       feature_name: str
       has_leakage: bool
       leakage_type: Optional[str]  # "future_data", "target_leakage", "timestamp"
       leakage_score: float  # 0-1, higher = more leakage risk
       details: str
```

4. **Integration:**
   - Call after feature generation, before training
   - Option to auto-remove features with leakage
   - Log results to experiment tracker

---

### 3.2 Feature Pipeline Robustness

**File:** `src/features/pipeline.py`

**Tasks:**

1. **Add Input Validation:**
   - Validate DataFrame has required columns (open, high, low, close, volume)
   - Validate no negative prices
   - Validate timestamps are sorted
   - Validate no duplicate timestamps

2. **Add NaN/Inf Handling:**
   - Replace inf with NaN before processing
   - Track NaN percentage per feature
   - Warn if NaN percentage > threshold (e.g., 5%)
   - Option to drop high-NaN features

3. **Add Feature Statistics Logging:**
   - Log mean, std, min, max for each feature
   - Log correlation matrix summary
   - Log feature count at each stage

4. **Add Memory Optimization:**
   - Downcast float64 to float32 where precision allows
   - Delete intermediate DataFrames
   - Use gc.collect() after large operations

---

### 3.3 Technical Indicators Enhancement

**File:** `src/features/technical/indicators.py`

**Tasks:**

1. **Add Missing Indicators:**
   - Ichimoku Cloud (if not present)
   - Keltner Channels
   - Donchian Channels
   - Elder Ray Index
   - Chaikin Money Flow
   - Williams %R
   - Ultimate Oscillator

2. **Add Multi-Timeframe Features:**
   - Calculate indicators on multiple timeframes (15min, 1h, 4h, daily)
   - Resample data appropriately
   - Align timestamps correctly (no future data!)

3. **Add Indicator Normalization:**
   - Z-score normalization option
   - Min-max scaling option
   - Percentile ranking option

4. **Add Indicator Combinations:**
   - RSI + MACD divergence detection
   - Moving average crossover signals
   - Bollinger Band squeeze detection

---

### 3.4 Microstructure Features Completion

**File:** `src/features/microstructure/`

**Tasks:**

1. **Verify All Modules Work:**
   - Test `order_flow_imbalance.py`
   - Test `vpin.py`
   - Test `kyle_lambda.py`
   - Test `roll_spread.py`
   - Test `order_book_dynamics.py`
   - Fix any import errors or calculation bugs

2. **Add Missing Features:**
   - Effective spread calculation
   - Realized spread calculation
   - Price impact per trade
   - Trade flow toxicity

3. **Add Feature Aggregations:**
   - Rolling mean/std of microstructure features
   - Percentile rankings
   - Change from previous period

---

### 3.5 Time-Based Features Enhancement

**File:** `src/features/transformers.py` → `TimeCyclicalEncoder`

**Tasks:**

1. **Add More Time Features:**
   - Minute of hour (sin/cos)
   - Day of month (sin/cos)
   - Week of year (sin/cos)
   - Quarter of year (sin/cos)

2. **Add Market Session Features:**
   - Is market open (binary)
   - Minutes since market open
   - Minutes until market close
   - Session identifier (morning, midday, afternoon, close)

3. **Add Event-Based Time Features:**
   - Days until next FOMC meeting
   - Days until earnings (if available)
   - Days until options expiration (monthly)
   - Is first/last trading day of month

---

### 3.6 Feature Selection Module

**New File:** `src/features/feature_selection.py`

**Purpose:** Automated feature selection to reduce dimensionality and improve model performance.

**Implementation Requirements:**

1. **FeatureSelector Class:**
```
   class FeatureSelector:
       - __init__(self, method, n_features, threshold)
       - fit(X, y) -> self
       - transform(X) -> X_selected
       - get_selected_features() -> List[str]
       - get_feature_scores() -> Dict[str, float]
```

2. **Selection Methods:**
   - **Variance Threshold:** Remove low-variance features
   - **Correlation Filter:** Remove highly correlated features (keep one)
   - **Mutual Information:** Select features with highest MI with target
   - **Feature Importance:** Use model-based importance (LightGBM)
   - **Recursive Feature Elimination:** Iteratively remove worst features

3. **Stability Selection:**
   - Run feature selection multiple times with bootstrap
   - Keep features selected consistently (>50% of runs)
   - More robust than single selection

4. **Integration:**
   - Can be used in pipeline after feature generation
   - Save selected feature list for inference
   - Log selection results

---

## 🟢 SECTION 4: MAIN.PY INTEGRATION UPDATES

### 4.1 Add New CLI Arguments

**File:** `main.py`

**Tasks:**

Add these new command-line arguments:
```
--validate-features    Run feature leakage check before training
--check-drift         Run drift detection on existing model
--feature-selection   Apply feature selection before training
--save-features       Save generated features to disk
--load-features       Load pre-generated features from disk
--dry-run            Validate everything without actual training
--resume             Resume training from checkpoint
--model-path         Path to existing model (for evaluation/drift check)
```

Update argument parser and add handling logic for each.

---

### 4.2 Improve Error Handling

**File:** `main.py`

**Tasks:**

1. **Wrap Main Sections in Try/Except:**
   - Data loading section
   - Feature generation section
   - Training section
   - Backtesting section
   - Report generation section

2. **Add Specific Exception Handling:**
   - `FileNotFoundError`: Data files missing
   - `ValueError`: Invalid configuration
   - `MemoryError`: Not enough RAM
   - `KeyboardInterrupt`: Graceful shutdown

3. **Add Cleanup on Error:**
   - Save partial results if possible
   - Release memory
   - Close file handles
   - Log error details

4. **Add Retry Logic:**
   - For transient errors (e.g., MLflow connection)
   - Configurable retry count and delay

---

### 4.3 Add Data Validation Before Training

**File:** `main.py`

**Tasks:**

Add comprehensive data validation step after loading:

1. **Data Quality Checks:**
   - No future timestamps
   - No duplicate timestamps
   - Chronological order
   - Price sanity (no negative, no extreme outliers)
   - Volume sanity
   - No large gaps in data

2. **Data Statistics Logging:**
   - Date range
   - Number of bars per symbol
   - Missing data percentage
   - Price range per symbol

3. **Fail Fast:**
   - If critical data issues, abort with clear error message
   - If minor issues, log warning and continue

---

### 4.4 Improve Model Saving and Loading

**File:** `main.py`

**Tasks:**

1. **Standardize Model Naming:**
```
   models/{model_type}_{timestamp}_{metric_value}.pkl
   Example: models/lightgbm_20241217_153045_sharpe_1.45.pkl

Save Model Metadata:

Create JSON file alongside model:



json   {
     "model_type": "lightgbm",
     "trained_at": "2024-12-17T15:30:45",
     "training_data_range": ["2020-01-01", "2024-06-30"],
     "n_samples": 150000,
     "n_features": 85,
     "feature_names": [...],
     "cv_score_mean": 0.032,
     "cv_score_std": 0.008,
     "best_params": {...},
     "config_snapshot": {...}
   }

Save Feature Pipeline:

Save fitted FeaturePipeline alongside model
Required for inference to use same scaling parameters
Use joblib or pickle


Add Model Loading Function:

Load model + metadata + feature pipeline together
Validate compatibility
Return ready-to-use predictor




4.5 Add Training Checkpointing
File: main.py
Tasks:

Checkpoint During Training:

Save model state every N epochs (for deep learning)
Save intermediate results during CV
Allow resume from checkpoint


Checkpoint Contents:

Model weights/state
Optimizer state
Current epoch/fold
Best metrics so far
Training history


Resume Logic:

Detect existing checkpoint
Load state
Continue from where left off
Log that training was resumed




🔵 SECTION 5: TESTING AND VALIDATION
5.1 Create Unit Tests for Fixes
Directory: tests/unit/
Tasks:
Create unit tests for each fix:

test_feature_pipeline.py:

Test generate_features is callable on FeaturePipeline instance
Test fit/transform separation
Test leakage prevention


test_model_factory.py:

Test create_model method exists and works
Test all model types can be created
Test parameter passing


test_microstructure.py:

Test OrderBookDynamics import
Test OBI calculation
Test WMP calculation


test_validation.py:

Test purge gap calculation
Test no index overlap in CV splits
Test temporal ordering




5.2 Create Integration Tests
Directory: tests/integration/
Tasks:

test_training_pipeline.py:

Test full pipeline: data → features → training → evaluation
Use small sample data
Verify no errors


test_feature_to_model.py:

Test features integrate correctly with models
Test feature shapes match model expectations
Test no NaN in model input


test_config_loading.py:

Test config files load correctly
Test config merge logic
Test config validation




5.3 Add Validation Scripts
Directory: scripts/
Tasks:

validate_no_leakage.py:

Script to verify no data leakage in features
Run on full dataset
Report any issues


validate_cv_splits.py:

Script to visualize and validate CV splits
Check purge gaps
Check temporal ordering


validate_model_consistency.py:

Script to verify model produces consistent results
Run inference multiple times
Check for randomness issues




🟣 SECTION 6: CONFIGURATION AND DOCUMENTATION
6.1 Synchronize All Config Files
Files:

config/ml_config.yaml
config/trading_config.yaml
config/feature_params.yaml
main.py (DEFAULT_CONFIG)

Tasks:

Create Config Schema:

Define expected types for each config field
Define valid ranges/values
Add validation function


Synchronize Values:

Ensure same defaults across all files
Remove duplicates
Clear hierarchy (base → environment-specific)


Add Config Documentation:

Comment each field with description
Include valid values/ranges
Include examples




6.2 Create README for Training
File: src/training/README.md
Contents:

Overview of training module
List of all classes and their purposes
Training workflow diagram
Configuration options
Usage examples
Troubleshooting guide


6.3 Create README for Features
File: src/features/README.md
Contents:

Overview of feature engineering module
List of all features with descriptions
Leakage prevention guidelines
Feature pipeline usage
Adding custom features guide
Performance considerations


📊 SECTION 7: PERFORMANCE OPTIMIZATION
7.1 Memory Optimization
Files: Various
Tasks:

Use Efficient Data Types:

Convert float64 to float32 where possible
Use categorical dtype for string columns
Use int32 instead of int64 where range allows


Process Data in Chunks:

For large datasets, process symbol by symbol
Clear memory between symbols
Use generators where possible


Profile Memory Usage:

Add memory profiling to training script
Identify memory bottlenecks
Log peak memory usage




7.2 Speed Optimization
Files: Various
Tasks:

Use Vectorized Operations:

Replace loops with numpy/pandas vectorized operations
Use numba for custom calculations if needed


Enable Parallel Processing:

Use joblib for parallel feature generation
Use multiple cores for model training
Configure n_jobs appropriately


Add Caching:

Cache expensive feature calculations
Use feature store for pre-computed features
Implement cache invalidation




⚠️ SECTION 8: PRIORITY ORDER
CRITICAL (Do Immediately):

Fix Feature Pipeline indentation error
Fix ModelFactory.create → create_model
Fix Microstructure all export
Fix Features init.py exports
Synchronize config files (purge_gap, max_lookback)

HIGH PRIORITY (This Week):
6. Complete Deep Learning training implementation
7. Fix TrainingResult object usage
8. Add data validation before training
9. Improve model saving with metadata
10. Add unit tests for all fixes
MEDIUM PRIORITY (Next Week):
11. Create Drift Detection module
12. Create Feature Leakage Checker
13. Create Training Pipeline orchestrator
14. Add feature selection module
15. Add integration tests
LOWER PRIORITY (Backlog):
16. Add all new CLI arguments
17. Create documentation files
18. Performance optimizations
19. Additional technical indicators
20. Multi-timeframe features

✅ SECTION 9: SUCCESS CRITERIA
After all tasks are complete, verify:

✅ python main.py --mode train runs without errors
✅ python main.py --mode backtest runs without errors
✅ All unit tests pass: pytest tests/unit/
✅ All integration tests pass: pytest tests/integration/
✅ Feature pipeline leakage check passes
✅ CV splits have no index overlap
✅ Purge gap is correctly calculated (≥ prediction_horizon + max_lookback)
✅ Models are saved with metadata JSON
✅ Deep learning models actually train (loss decreases)
✅ All imports work without errors
✅ Config files are synchronized
✅ No warnings about deprecated methods


🔧 SECTION 10: EXECUTION NOTES

Before Each Change:

Read the existing code carefully
Understand the current implementation
Identify dependencies


After Each Change:

Run relevant tests
Check imports work
Verify no regressions


Code Style:

Follow existing code style
Add docstrings to new functions/classes
Add type hints
Keep functions focused and small


Git Commits:

Make small, focused commits
Write clear commit messages
Reference task number in commit


If Stuck:

Log the issue clearly
Document what was tried
Move to next task if blocked