# AI Agent Pre-Training Instructions
## AlphaTrade System - Critical Improvements Before Training

---

# 🚨 PRIORITY 1: DATA QUALITY FIXES (Must Do)

## Task 1: Fix Trading Hours Contamination
The CSV data contains bars from 09:00 to 23:45+. This mixes regular hours with extended hours data which have completely different dynamics.

**Do this:**
- Analyze all CSV timestamps and identify timezone (likely UTC)
- Convert all timestamps to US/Eastern
- Filter to ONLY regular market hours: 09:30-16:00 ET
- Remove all pre-market, after-hours, and overnight data from training set
- Verify each trading day has exactly 26 bars (6.5 hours × 4 bars/hour)

**Why critical:** Extended hours data has 10-100x less volume, wider spreads, more noise. Training on mixed data will degrade model performance significantly.

---

## Task 2: Handle Volume Anomalies
Data shows volume spikes of 100x (e.g., 10,173 → 9,849,086 in consecutive bars).

**Do this:**
- Calculate rolling 20-bar volume statistics for each symbol
- Flag bars where volume > 5x rolling average as `volume_spike`
- Cross-reference spike dates with earnings calendar and FOMC dates
- Create binary flags: `is_earnings_window`, `is_fomc_day`, `is_opex_day`
- Add these flags as features, do NOT remove these bars

**Why critical:** Models need to know when unusual market conditions exist.

---

## Task 3: Validate OHLC Data Integrity
Data shows inconsistent price precision (2 vs 4 decimals) suggesting merged data sources.

**Do this:**
- Standardize all prices to 2 decimal places
- Validate OHLC relationships: High ≥ max(O,C), Low ≤ min(O,C)
- Fix any violations by adjusting High/Low to valid values
- Check for and handle any stock splits in the date range

**Why critical:** Invalid OHLC data will produce garbage technical indicators.

---

# 🎯 PRIORITY 2: LABELING CALIBRATION (Must Do)

## Task 4: Calibrate Triple Barrier Per Symbol
Current implementation uses fixed parameters for all 46 symbols. AAPL and a small-cap stock should not use same barriers.

**Do this:**
- Calculate 20-day ATR for each symbol
- Set profit target = 1.5 × ATR
- Set stop loss = 1.0 × ATR  
- Find optimal max_holding_period per symbol (analyze average time to barrier touch)
- Create `config/triple_barrier_params.yaml` with per-symbol parameters
- Add regime adjustment: widen barriers by 1.5x when VIX > 25

**Why critical:** Wrong barrier sizes cause label noise and poor model training.

---

## Task 5: Validate Label Quality
Before training, verify labels are usable.

**Do this:**
- Calculate class distribution per symbol (target: each class 25-40%)
- Check label autocorrelation (target: < 0.1)
- Verify barrier touch distribution: upper/lower/vertical
- If vertical barrier > 40%, increase holding period
- If class imbalance > 60/40, adjust barrier multipliers

**Why critical:** Imbalanced or autocorrelated labels will cause overfitting.

---

# ⚖️ PRIORITY 3: VALIDATION SETUP (Must Do)

## Task 6: Verify Embargo Prevents Leakage
Current embargo may be insufficient given feature lookbacks up to 200 periods.

**Do this:**
- Calculate max feature lookback across all features (likely 200 for SMA_200)
- Set embargo = max_lookback + 20 buffer = 220 bars minimum
- Convert to percentage: embargo_pct = 220 / train_size
- Verify PurgedKFoldCV embargo_pct ≥ 5%
- Test for leakage: ensure no training sample overlaps with test labels

**Why critical:** Information leakage = fake backtest results = real trading losses.

---

## Task 7: Reserve Holdout Data
Lock away data that will NEVER be seen during development.

**Do this:**
- Temporal holdout: Last 3 months of data → move to `data/holdout/`
- Symbol holdout: Pick 6 symbols (2 per sector) → exclude from training
- Stress period: 2022 bear market (Jan-Jun 2022) → mark as stress test only
- Create `holdout_manifest.json` with exact dates and symbols
- Block training pipeline from accessing holdout folder

**Why critical:** True out-of-sample testing requires untouched data.

---

# 📊 PRIORITY 4: FEATURE IMPROVEMENTS (Should Do)

## Task 8: Remove Redundant Features
200+ features likely contain many redundant ones (correlation > 0.95).

**Do this:**
- Compute full feature correlation matrix
- Identify clusters of features with > 0.95 correlation
- Keep only 1 feature per cluster (highest variance)
- Reduce to 60-80 final features
- Document removed features and reasoning

**Why critical:** Redundant features slow training and can cause instability.

---

## Task 9: Add Regime Awareness
Models need to know current market regime.

**Do this:**
- Add VIX regime feature: low (<15), normal (15-25), high (25-35), extreme (>35)
- Add trend regime: bull (price > SMA50 > SMA200), bear (opposite), sideways
- Add volatility regime from existing HMM detector
- Add days_since_regime_change feature
- Ensure regime features are NOT forward-looking

**Why critical:** Same signal means different things in different regimes.

---

# 🔧 PRIORITY 5: CONFIGURATION (Should Do)

## Task 10: Symbol-Specific Parameters
Current `symbols.yaml` uses estimated values.

**Do this:**
- Calculate actual average spread from data: (high-low)/close × 10000 bps
- Calculate average daily volume per symbol
- Calculate beta to SPY for each symbol
- Update `symbols.yaml` with real values
- Group symbols by volatility: low/medium/high

**Why critical:** Accurate transaction costs = realistic backtests.

---

# 📋 EXECUTION ORDER

```
Week 1: Tasks 1, 2, 3 (Data Quality)
Week 2: Tasks 4, 5 (Labeling)  
Week 3: Tasks 6, 7 (Validation)
Week 4: Tasks 8, 9, 10 (Features & Config)
```

---

# ✅ SUCCESS CRITERIA

Before training, verify:

| Check | Target |
|-------|--------|
| Regular hours only | 26 bars/day, no extended hours |
| OHLC valid | 0 violations |
| Label balance | Each class 25-40% |
| Label autocorr | < 0.1 |
| Embargo | ≥ 5% of training data |
| Holdout reserved | 3 months + 6 symbols locked |
| Features | < 80, no redundancy |
| Leakage test | PASS |

---

# ⚠️ DO NOT

- Do NOT start training until all Priority 1-3 tasks complete
- Do NOT use extended hours data for training
- Do NOT use default parameters for all symbols
- Do NOT skip embargo verification
- Do NOT touch holdout data during development

---

**Total estimated time: 3-4 weeks**
**Most critical: Tasks 1, 4, 6, 7**
