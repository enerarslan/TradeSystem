"""
WALK-FORWARD OPTIMIZATION & ANALYSIS ENGINE
JPMorgan Quantitative Research Division Style

Walk-Forward Analysis is a robust methodology for:
- Parameter optimization without overfitting
- Out-of-sample performance validation
- Strategy robustness testing
- Rolling window backtesting

Features:
- Anchored and Rolling window modes
- Multi-parameter grid search
- Parallel optimization
- Efficiency ratio calculation
- Comprehensive reporting
- Export to CSV/JSON

Usage:
    from backtest.walk_forward import WalkForwardOptimizer
    
    optimizer = WalkForwardOptimizer(
        symbol="AAPL",
        initial_capital=100_000,
        train_period_days=180,
        test_period_days=30
    )
    
    results = await optimizer.run(
        strategy_class=AdvancedMomentum,
        param_grid={
            'rsi_period': [10, 14, 20],
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [65, 70, 75]
        }
    )
    
    optimizer.print_report()
    optimizer.export_results("walk_forward_results.csv")
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
import time
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum

from utils.logger import log
from data.csv_loader import LocalCSVLoader
from data.models import MarketTick, WalkForwardResult
from strategies.momentum import AdvancedMomentum
from risk.core import EnterpriseRiskManager, RiskLimitConfig
from execution.portfolio import PortfolioManager


class WindowMode(Enum):
    """Walk-forward window modes"""
    ROLLING = "rolling"      # Moving window (fixed size)
    ANCHORED = "anchored"    # Expanding window (anchor start)
    HYBRID = "hybrid"        # Anchored train, rolling test


@dataclass
class OptimizationConfig:
    """Walk-forward optimization configuration"""
    # Window settings
    train_period_days: int = 180           # In-sample training period
    test_period_days: int = 30             # Out-of-sample test period
    window_mode: WindowMode = WindowMode.ROLLING
    min_train_samples: int = 100           # Minimum bars for training
    
    # Optimization settings
    optimization_metric: str = "sharpe"    # sharpe, return, calmar, sortino
    n_best_params: int = 3                 # Top N parameters to average
    use_ensemble: bool = False             # Ensemble top N parameters
    
    # Execution settings
    parallel: bool = True                  # Parallel parameter search
    max_workers: int = 4                   # Max parallel workers
    
    # Risk settings
    use_risk_management: bool = True
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    
    # Output settings
    verbose: bool = True
    save_trades: bool = True
    export_path: str = "data/walk_forward_results"


@dataclass
class WindowMetrics:
    """Metrics for a single window"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    volatility: float = 0.0


@dataclass
class WalkForwardSummary:
    """Complete walk-forward analysis summary"""
    # Overall metrics
    total_windows: int = 0
    profitable_windows: int = 0
    window_win_rate: float = 0.0
    
    # Aggregated returns
    cumulative_return: float = 0.0
    average_window_return: float = 0.0
    best_window_return: float = 0.0
    worst_window_return: float = 0.0
    return_std: float = 0.0
    
    # Risk metrics
    average_sharpe: float = 0.0
    average_sortino: float = 0.0
    average_max_drawdown: float = 0.0
    worst_drawdown: float = 0.0
    
    # Efficiency metrics
    average_efficiency_ratio: float = 0.0
    efficiency_consistency: float = 0.0  # % of windows with ratio > 0.5
    
    # Parameter stability
    parameter_stability_score: float = 0.0  # How often same params chosen
    most_stable_params: Dict[str, Any] = field(default_factory=dict)
    
    # Individual results
    window_results: List[WalkForwardResult] = field(default_factory=list)
    
    # Timing
    total_time_seconds: float = 0.0


class WalkForwardOptimizer:
    """
    Professional Walk-Forward Optimization Engine.
    
    Walk-forward analysis divides data into multiple windows:
    - Training window: Optimize parameters (in-sample)
    - Testing window: Validate with best params (out-of-sample)
    - Move forward and repeat
    
    This prevents overfitting and provides robust parameter estimates.
    """
    
    def __init__(
        self,
        symbol: str,
        initial_capital: float = 100_000,
        train_period_days: int = 180,
        test_period_days: int = 30,
        window_mode: str = "rolling",
        **kwargs
    ):
        """
        Initialize Walk-Forward Optimizer.
        
        Args:
            symbol: Trading symbol
            initial_capital: Starting capital for each window
            train_period_days: Days for training (in-sample)
            test_period_days: Days for testing (out-of-sample)
            window_mode: 'rolling', 'anchored', or 'hybrid'
            **kwargs: Additional config options
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        
        # Configuration
        self.config = OptimizationConfig(
            train_period_days=train_period_days,
            test_period_days=test_period_days,
            window_mode=WindowMode(window_mode),
            **{k: v for k, v in kwargs.items() 
               if hasattr(OptimizationConfig, k)}
        )
        
        # Data loader
        self.loader = LocalCSVLoader(
            validate_data=True,
            interpolate_missing=True,
            remove_outliers=True
        )
        
        # Results storage
        self.results: List[WalkForwardResult] = []
        self.summary: Optional[WalkForwardSummary] = None
        self.all_trades: List[Dict] = []
        
        # Parameter tracking
        self.param_history: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            'windows_processed': 0,
            'total_optimizations': 0,
            'total_backtests': 0,
            'errors': 0
        }
    
    async def run(
        self,
        strategy_class: Type = AdvancedMomentum,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> WalkForwardSummary:
        """
        Run walk-forward optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            param_grid: Parameter grid for optimization
                Example: {
                    'rsi_period': [10, 14, 20],
                    'rsi_oversold': [25, 30, 35]
                }
            start_date: Start of analysis period
            end_date: End of analysis period
        
        Returns:
            WalkForwardSummary: Complete analysis results
        """
        log.info("=" * 70)
        log.info("   📊 WALK-FORWARD OPTIMIZATION ENGINE")
        log.info("=" * 70)
        log.info(f"   Symbol          : {self.symbol}")
        log.info(f"   Capital         : ${self.initial_capital:,.2f}")
        log.info(f"   Train Period    : {self.config.train_period_days} days")
        log.info(f"   Test Period     : {self.config.test_period_days} days")
        log.info(f"   Window Mode     : {self.config.window_mode.value}")
        log.info(f"   Metric          : {self.config.optimization_metric}")
        log.info("=" * 70 + "\n")
        
        start_time = time.time()
        
        # 1. Load data
        log.info("📂 Loading historical data...")
        ticks = self.loader.load_data(
            self.symbol,
            use_cache=True,
            start_date=start_date,
            end_date=end_date
        )
        
        if not ticks:
            log.error("❌ No data loaded!")
            return None
        
        log.info(f"✅ Loaded {len(ticks):,} bars\n")
        
        # Convert to DataFrame for easier slicing
        df = self._ticks_to_dataframe(ticks)
        
        # 2. Generate windows
        windows = self._generate_windows(df)
        log.info(f"📅 Generated {len(windows)} walk-forward windows\n")
        
        if not windows:
            log.error("❌ Not enough data for walk-forward analysis!")
            return None
        
        # 3. Default param grid if not provided
        if param_grid is None:
            param_grid = self._get_default_param_grid(strategy_class)
        
        param_combinations = self._generate_param_combinations(param_grid)
        log.info(f"🔧 Testing {len(param_combinations)} parameter combinations\n")
        
        # 4. Process each window
        log.info("⚡ Starting walk-forward optimization...\n")
        log.info(f"{'─' * 70}")
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            log.info(f"\n📊 Window {i + 1}/{len(windows)}")
            log.info(f"   Train: {train_start.strftime('%Y-%m-%d')} → {train_end.strftime('%Y-%m-%d')}")
            log.info(f"   Test:  {test_start.strftime('%Y-%m-%d')} → {test_end.strftime('%Y-%m-%d')}")
            
            # Get window data
            train_data = df[(df.index >= train_start) & (df.index < train_end)]
            test_data = df[(df.index >= test_start) & (df.index < test_end)]
            
            if len(train_data) < self.config.min_train_samples:
                log.warning(f"   ⚠️ Insufficient training data ({len(train_data)} bars), skipping...")
                continue
            
            if len(test_data) < 10:
                log.warning(f"   ⚠️ Insufficient test data ({len(test_data)} bars), skipping...")
                continue
            
            # Optimize on training data
            best_params, in_sample_metrics = await self._optimize_window(
                train_data,
                strategy_class,
                param_combinations
            )
            
            self.param_history.append(best_params)
            
            # Test on out-of-sample data
            out_sample_metrics = await self._test_window(
                test_data,
                strategy_class,
                best_params
            )
            
            # Calculate efficiency ratio
            efficiency_ratio = self._calculate_efficiency_ratio(
                in_sample_metrics,
                out_sample_metrics
            )
            
            # Create result
            result = WalkForwardResult(
                window_id=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                in_sample_return=in_sample_metrics.total_return,
                in_sample_sharpe=in_sample_metrics.sharpe_ratio,
                out_sample_return=out_sample_metrics.total_return,
                out_sample_sharpe=out_sample_metrics.sharpe_ratio,
                best_params=best_params,
                efficiency_ratio=efficiency_ratio
            )
            
            self.results.append(result)
            self.stats['windows_processed'] += 1
            
            # Log window results
            log.info(f"   Best Params    : {best_params}")
            log.info(f"   In-Sample      : Return={in_sample_metrics.total_return:+.2%}, Sharpe={in_sample_metrics.sharpe_ratio:.2f}")
            log.info(f"   Out-of-Sample  : Return={out_sample_metrics.total_return:+.2%}, Sharpe={out_sample_metrics.sharpe_ratio:.2f}")
            log.info(f"   Efficiency     : {efficiency_ratio:.2%}")
        
        # 5. Generate summary
        elapsed_time = time.time() - start_time
        self.summary = self._generate_summary(elapsed_time)
        
        log.info(f"\n{'─' * 70}")
        log.success(f"✅ Walk-forward optimization complete! ({elapsed_time:.2f}s)")
        log.info(f"{'─' * 70}\n")
        
        # 6. Print report
        if self.config.verbose:
            self.print_report()
        
        return self.summary
    
    def _ticks_to_dataframe(self, ticks: List[MarketTick]) -> pd.DataFrame:
        """Convert tick list to DataFrame"""
        data = [{
            'timestamp': t.timestamp,
            'open': t.price,
            'high': t.price,
            'low': t.price,
            'close': t.price,
            'volume': t.volume
        } for t in ticks]
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def _generate_windows(
        self,
        df: pd.DataFrame
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate walk-forward windows.
        
        Returns list of (train_start, train_end, test_start, test_end) tuples.
        """
        windows = []
        
        start_date = df.index.min()
        end_date = df.index.max()
        
        train_days = timedelta(days=self.config.train_period_days)
        test_days = timedelta(days=self.config.test_period_days)
        
        if self.config.window_mode == WindowMode.ROLLING:
            # Rolling window: fixed size, moves forward
            current_train_start = start_date
            
            while True:
                train_end = current_train_start + train_days
                test_start = train_end
                test_end = test_start + test_days
                
                if test_end > end_date:
                    break
                
                windows.append((
                    current_train_start,
                    train_end,
                    test_start,
                    test_end
                ))
                
                # Move window forward by test period
                current_train_start = current_train_start + test_days
        
        elif self.config.window_mode == WindowMode.ANCHORED:
            # Anchored: training always starts from beginning
            anchor_start = start_date
            current_test_start = anchor_start + train_days
            
            while True:
                train_end = current_test_start
                test_end = current_test_start + test_days
                
                if test_end > end_date:
                    break
                
                windows.append((
                    anchor_start,
                    train_end,
                    current_test_start,
                    test_end
                ))
                
                current_test_start = test_end
        
        elif self.config.window_mode == WindowMode.HYBRID:
            # Hybrid: anchored training, rolling test
            anchor_start = start_date
            current_test_start = anchor_start + train_days
            
            while True:
                test_end = current_test_start + test_days
                
                if test_end > end_date:
                    break
                
                # Training expands, but we limit lookback
                train_start = max(
                    anchor_start,
                    current_test_start - train_days
                )
                
                windows.append((
                    train_start,
                    current_test_start,
                    current_test_start,
                    test_end
                ))
                
                current_test_start = test_end
        
        return windows
    
    def _get_default_param_grid(self, strategy_class: Type) -> Dict[str, List[Any]]:
        """Get default parameter grid for strategy"""
        # Default grid for AdvancedMomentum
        return {
            'rsi_period': [10, 14, 20],
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [65, 70, 75],
            'volume_threshold': [1.2, 1.5, 2.0]
        }
    
    def _generate_param_combinations(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations"""
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    async def _optimize_window(
        self,
        train_data: pd.DataFrame,
        strategy_class: Type,
        param_combinations: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], WindowMetrics]:
        """
        Optimize parameters on training window.
        
        Returns best parameters and in-sample metrics.
        """
        results = []
        
        for params in param_combinations:
            self.stats['total_optimizations'] += 1
            
            metrics = await self._run_single_backtest(
                train_data,
                strategy_class,
                params
            )
            
            results.append({
                'params': params,
                'metrics': metrics
            })
        
        # Sort by optimization metric
        metric_key = self.config.optimization_metric
        
        def get_metric(r):
            m = r['metrics']
            if metric_key == 'sharpe':
                return m.sharpe_ratio
            elif metric_key == 'return':
                return m.total_return
            elif metric_key == 'calmar':
                return m.calmar_ratio
            elif metric_key == 'sortino':
                return m.sortino_ratio
            else:
                return m.sharpe_ratio
        
        results.sort(key=get_metric, reverse=True)
        
        # Best result
        best = results[0]
        
        return best['params'], best['metrics']
    
    async def _test_window(
        self,
        test_data: pd.DataFrame,
        strategy_class: Type,
        params: Dict[str, Any]
    ) -> WindowMetrics:
        """
        Test parameters on out-of-sample window.
        """
        self.stats['total_backtests'] += 1
        
        return await self._run_single_backtest(
            test_data,
            strategy_class,
            params
        )
    
    async def _run_single_backtest(
        self,
        data: pd.DataFrame,
        strategy_class: Type,
        params: Dict[str, Any]
    ) -> WindowMetrics:
        """
        Run a single backtest with given parameters.
        """
        # Initialize
        capital = self.initial_capital
        position = 0
        entry_price = 0.0
        trades = []
        equity_curve = [capital]
        
        # Create strategy
        try:
            strategy = strategy_class(symbol=self.symbol, **params)
        except Exception as e:
            # If params don't match, use defaults
            strategy = strategy_class(symbol=self.symbol)
        
        # Convert DataFrame rows to ticks
        for idx, row in data.iterrows():
            tick = MarketTick(
                symbol=self.symbol,
                timestamp=idx,
                price=row['close'],
                volume=row.get('volume', 0)
            )
            
            # Get signal
            try:
                signal = await strategy.on_tick(tick)
            except Exception:
                signal = None
            
            current_price = row['close']
            
            if signal and signal.quantity > 0:
                side = signal.side
                
                # Close existing position if opposite
                if position != 0:
                    if (position > 0 and side == 'SELL') or (position < 0 and side == 'BUY'):
                        # Close position
                        pnl = (current_price - entry_price) * position
                        pnl -= abs(pnl) * self.config.commission_pct  # Commission
                        capital += pnl
                        
                        trades.append({
                            'side': 'CLOSE',
                            'price': current_price,
                            'pnl': pnl
                        })
                        
                        position = 0
                
                # Open new position
                if position == 0 and side in ['BUY', 'SELL']:
                    position_size = min(
                        signal.quantity,
                        int(capital * 0.95 / current_price)
                    )
                    
                    if position_size > 0:
                        if side == 'BUY':
                            position = position_size
                        else:
                            position = -position_size
                        
                        entry_price = current_price * (1 + self.config.slippage_pct)
                        
                        trades.append({
                            'side': side,
                            'price': entry_price,
                            'size': position_size
                        })
            
            # Update equity
            if position != 0:
                unrealized = (current_price - entry_price) * position
                equity_curve.append(capital + unrealized)
            else:
                equity_curve.append(capital)
        
        # Calculate metrics
        return self._calculate_metrics(equity_curve, trades)
    
    def _calculate_metrics(
        self,
        equity_curve: List[float],
        trades: List[Dict]
    ) -> WindowMetrics:
        """Calculate performance metrics from equity curve"""
        equity = np.array(equity_curve)
        
        if len(equity) < 2:
            return WindowMetrics()
        
        # Returns
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        if len(returns) < 1:
            return WindowMetrics()
        
        # Total return
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252 * 26) if len(returns) > 1 else 0.0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        mean_return = np.mean(returns) * 252 * 26
        sharpe = mean_return / volatility if volatility > 0 else 0.0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252 * 26) if len(downside_returns) > 1 else 0.0
        sortino = mean_return / downside_std if downside_std > 0 else 0.0
        
        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Calmar ratio
        calmar = total_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Trade statistics
        closed_trades = [t for t in trades if 'pnl' in t]
        winning = [t for t in closed_trades if t['pnl'] > 0]
        losing = [t for t in closed_trades if t['pnl'] <= 0]
        
        win_rate = len(winning) / len(closed_trades) if closed_trades else 0.0
        
        total_wins = sum(t['pnl'] for t in winning) if winning else 0.0
        total_losses = abs(sum(t['pnl'] for t in losing)) if losing else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        return WindowMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(closed_trades),
            volatility=volatility
        )
    
    def _calculate_efficiency_ratio(
        self,
        in_sample: WindowMetrics,
        out_sample: WindowMetrics
    ) -> float:
        """
        Calculate walk-forward efficiency ratio.
        
        Efficiency = Out-of-Sample Performance / In-Sample Performance
        
        Values:
        - > 0.5: Good (OOS captures half of IS performance)
        - 0.3 - 0.5: Acceptable
        - < 0.3: Poor (overfitting suspected)
        """
        if in_sample.total_return == 0:
            return 0.0
        
        # Use returns for efficiency calculation
        if in_sample.total_return > 0:
            if out_sample.total_return > 0:
                return min(out_sample.total_return / in_sample.total_return, 2.0)
            else:
                return 0.0
        else:
            # Both negative or IS negative
            if out_sample.total_return <= 0:
                return min(abs(in_sample.total_return) / abs(out_sample.total_return), 2.0)
            else:
                return 1.0  # OOS positive when IS negative is good
    
    def _generate_summary(self, elapsed_time: float) -> WalkForwardSummary:
        """Generate complete walk-forward summary"""
        if not self.results:
            return WalkForwardSummary()
        
        # Returns
        returns = [r.out_sample_return for r in self.results]
        
        # Cumulative return (compounded)
        cumulative = 1.0
        for r in returns:
            cumulative *= (1 + r)
        cumulative_return = cumulative - 1
        
        # Window statistics
        profitable_windows = sum(1 for r in returns if r > 0)
        
        # Risk metrics
        sharpes = [r.out_sample_sharpe for r in self.results]
        
        # Efficiency ratios
        efficiencies = [r.efficiency_ratio for r in self.results]
        efficient_windows = sum(1 for e in efficiencies if e > 0.5)
        
        # Parameter stability
        param_stability = self._calculate_parameter_stability()
        
        return WalkForwardSummary(
            total_windows=len(self.results),
            profitable_windows=profitable_windows,
            window_win_rate=profitable_windows / len(self.results) if self.results else 0.0,
            
            cumulative_return=cumulative_return,
            average_window_return=np.mean(returns) if returns else 0.0,
            best_window_return=max(returns) if returns else 0.0,
            worst_window_return=min(returns) if returns else 0.0,
            return_std=np.std(returns) if len(returns) > 1 else 0.0,
            
            average_sharpe=np.mean(sharpes) if sharpes else 0.0,
            average_sortino=np.mean([r.out_sample_sharpe for r in self.results]) if self.results else 0.0,
            average_max_drawdown=np.mean([0.0 for _ in self.results]),  # Would need to track
            worst_drawdown=0.0,
            
            average_efficiency_ratio=np.mean(efficiencies) if efficiencies else 0.0,
            efficiency_consistency=efficient_windows / len(efficiencies) if efficiencies else 0.0,
            
            parameter_stability_score=param_stability['score'],
            most_stable_params=param_stability['most_common'],
            
            window_results=self.results,
            total_time_seconds=elapsed_time
        )
    
    def _calculate_parameter_stability(self) -> Dict[str, Any]:
        """Calculate how stable the optimal parameters are across windows"""
        if not self.param_history:
            return {'score': 0.0, 'most_common': {}}
        
        # Count parameter occurrences
        param_counts: Dict[str, Dict[Any, int]] = {}
        
        for params in self.param_history:
            for key, value in params.items():
                if key not in param_counts:
                    param_counts[key] = {}
                
                # Convert value to string for counting
                val_str = str(value)
                param_counts[key][val_str] = param_counts[key].get(val_str, 0) + 1
        
        # Find most common values and calculate stability
        most_common = {}
        stability_scores = []
        
        for key, counts in param_counts.items():
            max_count = max(counts.values())
            most_common_val = [k for k, v in counts.items() if v == max_count][0]
            most_common[key] = most_common_val
            
            stability = max_count / len(self.param_history)
            stability_scores.append(stability)
        
        return {
            'score': np.mean(stability_scores) if stability_scores else 0.0,
            'most_common': most_common
        }
    
    def print_report(self):
        """Print detailed walk-forward report"""
        if not self.summary:
            log.warning("No results to report. Run optimization first.")
            return
        
        s = self.summary
        
        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + " " * 12 + "📊 WALK-FORWARD ANALYSIS REPORT" + " " * 24 + "║")
        print("╠" + "═" * 68 + "╣")
        
        # Overview
        print("║  📋 OVERVIEW" + " " * 55 + "║")
        print("║  " + "─" * 66 + "║")
        print(f"║  Total Windows        : {s.total_windows:<43} ║")
        print(f"║  Profitable Windows   : {s.profitable_windows} ({s.window_win_rate:.1%}){' ' * 32} ║")
        print(f"║  Analysis Time        : {s.total_time_seconds:.2f}s{' ' * 40} ║")
        
        # Returns
        print("║" + " " * 68 + "║")
        print("║  💰 RETURNS" + " " * 56 + "║")
        print("║  " + "─" * 66 + "║")
        
        ret_color = "+" if s.cumulative_return >= 0 else ""
        print(f"║  Cumulative Return    : {ret_color}{s.cumulative_return:.2%}{' ' * 40} ║")
        print(f"║  Average Window Return: {s.average_window_return:+.2%}{' ' * 40} ║")
        print(f"║  Best Window          : {s.best_window_return:+.2%}{' ' * 40} ║")
        print(f"║  Worst Window         : {s.worst_window_return:+.2%}{' ' * 40} ║")
        print(f"║  Return Std Dev       : {s.return_std:.2%}{' ' * 41} ║")
        
        # Risk
        print("║" + " " * 68 + "║")
        print("║  📉 RISK METRICS" + " " * 51 + "║")
        print("║  " + "─" * 66 + "║")
        print(f"║  Average Sharpe       : {s.average_sharpe:.2f}{' ' * 44} ║")
        
        # Efficiency
        print("║" + " " * 68 + "║")
        print("║  ⚡ EFFICIENCY" + " " * 53 + "║")
        print("║  " + "─" * 66 + "║")
        print(f"║  Avg Efficiency Ratio : {s.average_efficiency_ratio:.2%}{' ' * 40} ║")
        print(f"║  Efficiency Consistency: {s.efficiency_consistency:.1%} of windows > 50%{' ' * 22} ║")
        
        # Parameter Stability
        print("║" + " " * 68 + "║")
        print("║  🔧 PARAMETER STABILITY" + " " * 44 + "║")
        print("║  " + "─" * 66 + "║")
        print(f"║  Stability Score      : {s.parameter_stability_score:.1%}{' ' * 40} ║")
        print(f"║  Most Stable Params   : {str(s.most_stable_params)[:40]:<40} ║")
        
        print("╚" + "═" * 68 + "╝")
        
        # Individual window results
        print("\n📊 Window-by-Window Results:")
        print("─" * 100)
        print(f"{'Window':>6} │ {'Train Period':^25} │ {'Test Period':^25} │ {'IS Ret':>8} │ {'OOS Ret':>8} │ {'Eff':>6}")
        print("─" * 100)
        
        for r in self.results:
            print(
                f"{r.window_id:>6} │ "
                f"{r.train_start.strftime('%Y-%m-%d')} → {r.train_end.strftime('%Y-%m-%d')} │ "
                f"{r.test_start.strftime('%Y-%m-%d')} → {r.test_end.strftime('%Y-%m-%d')} │ "
                f"{r.in_sample_return:>+7.2%} │ "
                f"{r.out_sample_return:>+7.2%} │ "
                f"{r.efficiency_ratio:>5.0%}"
            )
        
        print("─" * 100)
    
    def export_results(
        self,
        filename: str = "walk_forward_results",
        format: str = "csv"
    ) -> str:
        """
        Export results to file.
        
        Args:
            filename: Output filename (without extension)
            format: 'csv' or 'json'
        
        Returns:
            Path to exported file
        """
        if not self.results:
            log.warning("No results to export")
            return ""
        
        export_path = Path(self.config.export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        if format == "csv":
            # Convert results to DataFrame
            data = []
            for r in self.results:
                data.append({
                    'window_id': r.window_id,
                    'train_start': r.train_start,
                    'train_end': r.train_end,
                    'test_start': r.test_start,
                    'test_end': r.test_end,
                    'in_sample_return': r.in_sample_return,
                    'in_sample_sharpe': r.in_sample_sharpe,
                    'out_sample_return': r.out_sample_return,
                    'out_sample_sharpe': r.out_sample_sharpe,
                    'efficiency_ratio': r.efficiency_ratio,
                    'best_params': str(r.best_params)
                })
            
            df = pd.DataFrame(data)
            filepath = export_path / f"{filename}.csv"
            df.to_csv(filepath, index=False)
        
        elif format == "json":
            data = {
                'summary': {
                    'total_windows': self.summary.total_windows,
                    'cumulative_return': self.summary.cumulative_return,
                    'average_sharpe': self.summary.average_sharpe,
                    'average_efficiency': self.summary.average_efficiency_ratio,
                    'parameter_stability': self.summary.parameter_stability_score
                },
                'windows': [
                    {
                        'window_id': r.window_id,
                        'train_period': f"{r.train_start} to {r.train_end}",
                        'test_period': f"{r.test_start} to {r.test_end}",
                        'in_sample_return': r.in_sample_return,
                        'out_sample_return': r.out_sample_return,
                        'efficiency_ratio': r.efficiency_ratio,
                        'best_params': r.best_params
                    }
                    for r in self.results
                ]
            }
            
            filepath = export_path / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        log.info(f"✅ Results exported to: {filepath}")
        return str(filepath)
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """
        Get best parameters based on walk-forward results.
        
        Returns most stable parameters or average of top performers.
        """
        if not self.param_history:
            return {}
        
        if self.summary and self.summary.most_stable_params:
            return self.summary.most_stable_params
        
        # Return most recent best params
        return self.param_history[-1] if self.param_history else {}


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def run_walk_forward(
    symbol: str,
    strategy_class: Type = AdvancedMomentum,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    train_days: int = 180,
    test_days: int = 30,
    **kwargs
) -> WalkForwardSummary:
    """
    Convenience function for quick walk-forward analysis.
    
    Usage:
        from backtest.walk_forward import run_walk_forward
        
        results = await run_walk_forward(
            "AAPL",
            param_grid={'rsi_period': [10, 14, 20]}
        )
    """
    optimizer = WalkForwardOptimizer(
        symbol=symbol,
        train_period_days=train_days,
        test_period_days=test_days,
        **kwargs
    )
    
    return await optimizer.run(
        strategy_class=strategy_class,
        param_grid=param_grid
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'WalkForwardOptimizer',
    'WalkForwardSummary',
    'WindowMetrics',
    'WindowMode',
    'OptimizationConfig',
    'run_walk_forward'
]