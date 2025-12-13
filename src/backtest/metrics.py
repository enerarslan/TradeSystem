"""
Performance Metrics and Reporting
JPMorgan-Level Performance Attribution and Risk Analytics

Features:
- Comprehensive performance metrics
- Risk-adjusted returns
- Trade analysis
- Performance attribution
- Report generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import scipy.stats as stats
from collections import defaultdict

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Returns
    total_return: float
    annualized_return: float
    cumulative_return: float
    ytd_return: float
    mtd_return: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: float

    # Risk
    volatility: float
    downside_deviation: float
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int  # Days

    # Distribution
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float

    # Stability
    stability: float  # R-squared of returns regression
    tail_ratio: float  # 95th percentile / 5th percentile

    def to_dict(self) -> Dict[str, float]:
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis
        }


@dataclass
class RiskMetrics:
    """Risk-specific metrics"""
    # Value at Risk
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    # Drawdown
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    current_drawdown: float

    # Volatility
    realized_volatility: float
    implied_volatility: Optional[float]
    volatility_of_volatility: float

    # Beta and correlation
    beta: float
    correlation_to_market: float
    r_squared: float

    # Tail risk
    left_tail_ratio: float
    right_tail_ratio: float
    gain_to_pain_ratio: float


@dataclass
class TradeMetrics:
    """Trade analysis metrics"""
    # Counts
    total_trades: int
    winning_trades: int
    losing_trades: int
    long_trades: int
    short_trades: int

    # Win/Loss
    win_rate: float
    loss_rate: float
    profit_factor: float
    payoff_ratio: float

    # P&L
    total_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Duration
    avg_trade_duration: float  # Hours
    avg_winning_duration: float
    avg_losing_duration: float

    # Streaks
    max_consecutive_wins: int
    max_consecutive_losses: int
    current_streak: int

    # Costs
    total_commission: float
    total_slippage: float
    avg_commission_per_trade: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_pnl': self.avg_pnl,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses
        }


class MetricsCalculator:
    """
    Comprehensive metrics calculation engine.

    Calculates performance, risk, and trade metrics
    following institutional standards.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
        benchmark_returns: Optional[pd.Series] = None
    ):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.benchmark_returns = benchmark_returns

    def calculate_performance_metrics(
        self,
        returns: pd.Series,
        equity_curve: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0:
            return self._empty_performance_metrics()

        # Basic returns
        total_return = (1 + returns).prod() - 1
        ann_return = self._annualize_return(total_return, len(returns))
        cumulative = (1 + returns).cumprod()

        # YTD and MTD
        ytd_return = self._calculate_ytd_return(returns)
        mtd_return = self._calculate_mtd_return(returns)

        # Volatility
        volatility = returns.std() * np.sqrt(self.periods_per_year)
        downside = returns[returns < 0]
        downside_deviation = downside.std() * np.sqrt(self.periods_per_year) if len(downside) > 0 else 0

        # Risk-adjusted ratios
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)

        # Drawdown
        if equity_curve is not None:
            dd_series = self._calculate_drawdown_series(equity_curve)
        else:
            dd_series = self._calculate_drawdown_series(cumulative)

        max_dd = dd_series.max()
        avg_dd = dd_series[dd_series > 0].mean() if (dd_series > 0).any() else 0
        max_dd_duration = self._calculate_max_drawdown_duration(dd_series)

        calmar = ann_return / max_dd if max_dd > 0 else 0

        # Omega ratio
        omega = self._calculate_omega(returns)

        # Information ratio
        info_ratio = self._calculate_information_ratio(returns)

        # Distribution metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # VaR
        var_95 = self._calculate_var(returns, 0.95)
        cvar_95 = self._calculate_cvar(returns, 0.95)

        # Stability
        stability = self._calculate_stability(returns)
        tail_ratio = self._calculate_tail_ratio(returns)

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=ann_return,
            cumulative_return=cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0,
            ytd_return=ytd_return,
            mtd_return=mtd_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            information_ratio=info_ratio,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration=max_dd_duration,
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            stability=stability,
            tail_ratio=tail_ratio
        )

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        equity_curve: Optional[pd.Series] = None,
        market_returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        if len(returns) == 0:
            return self._empty_risk_metrics()

        # VaR calculations
        var_95 = self._calculate_var(returns, 0.95)
        var_99 = self._calculate_var(returns, 0.99)
        cvar_95 = self._calculate_cvar(returns, 0.95)
        cvar_99 = self._calculate_cvar(returns, 0.99)

        # Drawdown
        cumulative = (1 + returns).cumprod()
        dd_series = self._calculate_drawdown_series(
            equity_curve if equity_curve is not None else cumulative
        )

        max_dd = dd_series.max()
        avg_dd = dd_series[dd_series > 0].mean() if (dd_series > 0).any() else 0
        max_dd_duration = self._calculate_max_drawdown_duration(dd_series)
        current_dd = dd_series.iloc[-1] if len(dd_series) > 0 else 0

        # Volatility
        realized_vol = returns.std() * np.sqrt(self.periods_per_year)

        # Volatility of volatility
        rolling_vol = returns.rolling(21).std()
        vol_of_vol = rolling_vol.std() * np.sqrt(self.periods_per_year) if len(rolling_vol.dropna()) > 0 else 0

        # Beta and correlation
        if market_returns is not None and len(market_returns) > 0:
            aligned = pd.concat([returns, market_returns], axis=1).dropna()
            if len(aligned) > 0:
                cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
                correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                r_squared = correlation ** 2
            else:
                beta, correlation, r_squared = 1.0, 0.0, 0.0
        else:
            beta, correlation, r_squared = 1.0, 0.0, 0.0

        # Tail risk
        left_tail = self._calculate_tail_ratio(returns, left=True)
        right_tail = self._calculate_tail_ratio(returns, left=False)

        # Gain to pain ratio
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        gain_to_pain = gains / losses if losses > 0 else float('inf')

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration=max_dd_duration,
            current_drawdown=current_dd,
            realized_volatility=realized_vol,
            implied_volatility=None,
            volatility_of_volatility=vol_of_vol,
            beta=beta,
            correlation_to_market=correlation,
            r_squared=r_squared,
            left_tail_ratio=left_tail,
            right_tail_ratio=right_tail,
            gain_to_pain_ratio=gain_to_pain
        )

    def calculate_trade_metrics(
        self,
        trades: List[Dict],
        initial_capital: float = 1000000
    ) -> TradeMetrics:
        """Calculate comprehensive trade metrics"""
        if not trades:
            return self._empty_trade_metrics()

        # Separate by P&L
        pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
        if not pnls:
            pnls = [0]

        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        # Counts
        total = len(trades)
        winning = len(winners)
        losing = len(losers)

        long_trades = len([t for t in trades if t.get('side') == 'buy'])
        short_trades = len([t for t in trades if t.get('side') == 'sell'])

        # Win/Loss rates
        win_rate = winning / total if total > 0 else 0
        loss_rate = losing / total if total > 0 else 0

        # Profit factor
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # P&L stats
        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls) if pnls else 0
        avg_win = np.mean(winners) if winners else 0
        avg_loss = np.mean(losers) if losers else 0
        largest_win = max(winners) if winners else 0
        largest_loss = min(losers) if losers else 0

        # Payoff ratio
        payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Duration
        durations = []
        win_durations = []
        lose_durations = []

        for t in trades:
            if 'entry_time' in t and 'exit_time' in t:
                duration = (t['exit_time'] - t['entry_time']).total_seconds() / 3600
                durations.append(duration)
                if t.get('pnl', 0) > 0:
                    win_durations.append(duration)
                else:
                    lose_durations.append(duration)

        avg_duration = np.mean(durations) if durations else 0
        avg_win_duration = np.mean(win_durations) if win_durations else 0
        avg_lose_duration = np.mean(lose_durations) if lose_durations else 0

        # Streaks
        max_wins, max_losses, current = self._calculate_streaks(pnls)

        # Costs
        total_commission = sum(t.get('commission', 0) for t in trades)
        total_slippage = sum(t.get('slippage', 0) for t in trades)
        avg_commission = total_commission / total if total > 0 else 0

        return TradeMetrics(
            total_trades=total,
            winning_trades=winning,
            losing_trades=losing,
            long_trades=long_trades,
            short_trades=short_trades,
            win_rate=win_rate,
            loss_rate=loss_rate,
            profit_factor=profit_factor,
            payoff_ratio=payoff,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_duration,
            avg_winning_duration=avg_win_duration,
            avg_losing_duration=avg_lose_duration,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            current_streak=current,
            total_commission=total_commission,
            total_slippage=total_slippage,
            avg_commission_per_trade=avg_commission
        )

    def _annualize_return(
        self,
        total_return: float,
        periods: int
    ) -> float:
        """Annualize total return"""
        if periods == 0:
            return 0
        years = periods / self.periods_per_year
        if years == 0:
            return 0
        return (1 + total_return) ** (1 / years) - 1

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0

        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        return np.sqrt(self.periods_per_year) * excess_returns.mean() / returns.std()

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0

        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        downside = returns[returns < 0]

        if len(downside) == 0 or downside.std() == 0:
            return 0

        return np.sqrt(self.periods_per_year) * excess_returns.mean() / downside.std()

    def _calculate_omega(
        self,
        returns: pd.Series,
        threshold: float = 0
    ) -> float:
        """Calculate Omega ratio"""
        if len(returns) == 0:
            return 0

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]

        if losses.sum() == 0:
            return float('inf')

        return gains.sum() / losses.sum()

    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate Information ratio vs benchmark"""
        if self.benchmark_returns is None or len(returns) == 0:
            return 0

        aligned = pd.concat([returns, self.benchmark_returns], axis=1).dropna()
        if len(aligned) == 0:
            return 0

        excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        tracking_error = excess.std()

        if tracking_error == 0:
            return 0

        return np.sqrt(self.periods_per_year) * excess.mean() / tracking_error

    def _calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0
        return -np.percentile(returns, (1 - confidence) * 100)

    def _calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        if len(returns) == 0:
            return 0

        var = self._calculate_var(returns, confidence)
        return -returns[returns <= -var].mean() if len(returns[returns <= -var]) > 0 else var

    def _calculate_drawdown_series(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        cum_max = equity.cummax()
        drawdown = (cum_max - equity) / cum_max
        return drawdown.fillna(0)

    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods"""
        if len(drawdown) == 0:
            return 0

        in_drawdown = drawdown > 0
        durations = []
        current_duration = 0

        for val in in_drawdown:
            if val:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return max(durations) if durations else 0

    def _calculate_stability(self, returns: pd.Series) -> float:
        """Calculate stability (R-squared of cumulative returns)"""
        if len(returns) < 2:
            return 0

        cumulative = (1 + returns).cumprod()
        x = np.arange(len(cumulative))

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, cumulative)
            return r_value ** 2
        except:
            return 0

    def _calculate_tail_ratio(
        self,
        returns: pd.Series,
        left: bool = None
    ) -> float:
        """Calculate tail ratio"""
        if len(returns) == 0:
            return 0

        if left is None:
            # Standard tail ratio
            right_tail = np.percentile(returns, 95)
            left_tail = np.percentile(returns, 5)
            return abs(right_tail / left_tail) if left_tail != 0 else 0
        elif left:
            return abs(np.percentile(returns, 5))
        else:
            return np.percentile(returns, 95)

    def _calculate_ytd_return(self, returns: pd.Series) -> float:
        """Calculate year-to-date return"""
        if len(returns) == 0:
            return 0

        current_year = returns.index[-1].year
        ytd_returns = returns[returns.index.year == current_year]
        return (1 + ytd_returns).prod() - 1 if len(ytd_returns) > 0 else 0

    def _calculate_mtd_return(self, returns: pd.Series) -> float:
        """Calculate month-to-date return"""
        if len(returns) == 0:
            return 0

        current_month = returns.index[-1].month
        current_year = returns.index[-1].year
        mtd_returns = returns[
            (returns.index.month == current_month) &
            (returns.index.year == current_year)
        ]
        return (1 + mtd_returns).prod() - 1 if len(mtd_returns) > 0 else 0

    def _calculate_streaks(
        self,
        pnls: List[float]
    ) -> Tuple[int, int, int]:
        """Calculate win/loss streaks"""
        if not pnls:
            return 0, 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        current_streak = 0

        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        # Current streak
        if pnls[-1] > 0:
            current_streak = current_wins
        elif pnls[-1] < 0:
            current_streak = -current_losses
        else:
            current_streak = 0

        return max_wins, max_losses, current_streak

    def _empty_performance_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics"""
        return PerformanceMetrics(
            total_return=0, annualized_return=0, cumulative_return=0,
            ytd_return=0, mtd_return=0, sharpe_ratio=0, sortino_ratio=0,
            calmar_ratio=0, omega_ratio=0, information_ratio=0,
            volatility=0, downside_deviation=0, max_drawdown=0,
            avg_drawdown=0, max_drawdown_duration=0, skewness=0,
            kurtosis=0, var_95=0, cvar_95=0, stability=0, tail_ratio=0
        )

    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics"""
        return RiskMetrics(
            var_95=0, var_99=0, cvar_95=0, cvar_99=0,
            max_drawdown=0, avg_drawdown=0, max_drawdown_duration=0,
            current_drawdown=0, realized_volatility=0, implied_volatility=None,
            volatility_of_volatility=0, beta=1.0, correlation_to_market=0,
            r_squared=0, left_tail_ratio=0, right_tail_ratio=0,
            gain_to_pain_ratio=0
        )

    def _empty_trade_metrics(self) -> TradeMetrics:
        """Return empty trade metrics"""
        return TradeMetrics(
            total_trades=0, winning_trades=0, losing_trades=0,
            long_trades=0, short_trades=0, win_rate=0, loss_rate=0,
            profit_factor=0, payoff_ratio=0, total_pnl=0, avg_pnl=0,
            avg_win=0, avg_loss=0, largest_win=0, largest_loss=0,
            avg_trade_duration=0, avg_winning_duration=0,
            avg_losing_duration=0, max_consecutive_wins=0,
            max_consecutive_losses=0, current_streak=0,
            total_commission=0, total_slippage=0, avg_commission_per_trade=0
        )


class PerformanceAttribution:
    """
    Performance attribution analysis.

    Breaks down returns by:
    - Factor attribution
    - Sector attribution
    - Security selection
    - Asset allocation
    """

    def __init__(
        self,
        benchmark_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None
    ):
        self.benchmark_returns = benchmark_returns
        self.factor_returns = factor_returns

    def brinson_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        benchmark_returns: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Brinson-Fachler attribution.

        Decomposes excess returns into:
        - Allocation effect
        - Selection effect
        - Interaction effect
        """
        # Align data
        aligned = self._align_attribution_data(
            portfolio_weights, portfolio_returns,
            benchmark_weights, benchmark_returns
        )

        if aligned is None:
            return {}

        pw, pr, bw, br = aligned

        # Calculate effects
        allocation = (pw - bw) * (br - br.mean())
        selection = bw * (pr - br)
        interaction = (pw - bw) * (pr - br)

        return {
            'allocation': allocation.sum(axis=1),
            'selection': selection.sum(axis=1),
            'interaction': interaction.sum(axis=1),
            'total_excess': allocation.sum(axis=1) + selection.sum(axis=1) + interaction.sum(axis=1)
        }

    def factor_attribution(
        self,
        returns: pd.Series,
        exposures: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Factor-based attribution.

        Attributes returns to factor exposures.
        """
        if self.factor_returns is None:
            return {}

        # Align data
        aligned = pd.concat([returns, self.factor_returns], axis=1).dropna()

        if len(aligned) == 0:
            return {}

        port_returns = aligned.iloc[:, 0]
        factors = aligned.iloc[:, 1:]

        # Regression
        from scipy.stats import linregress

        attributions = {}
        residual = port_returns.copy()

        for factor in factors.columns:
            slope, intercept, r, p, se = linregress(factors[factor], port_returns)
            contribution = slope * factors[factor]
            attributions[factor] = {
                'beta': slope,
                'contribution': contribution.sum(),
                'avg_contribution': contribution.mean()
            }
            residual = residual - contribution

        attributions['alpha'] = residual.mean() * 252
        attributions['residual_contribution'] = residual.sum()

        return attributions

    def sector_attribution(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        sector_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Sector-level attribution.

        Attributes returns by sector.
        """
        sector_returns = defaultdict(list)

        for date in positions.index:
            for symbol in positions.columns:
                if symbol in returns.columns:
                    sector = sector_map.get(symbol, 'Other')
                    weight = positions.loc[date, symbol]
                    ret = returns.loc[date, symbol] if date in returns.index else 0
                    sector_returns[sector].append({
                        'date': date,
                        'contribution': weight * ret
                    })

        # Aggregate by sector
        result = {}
        for sector, data in sector_returns.items():
            df = pd.DataFrame(data)
            if not df.empty:
                result[sector] = df.groupby('date')['contribution'].sum()

        return pd.DataFrame(result)

    def _align_attribution_data(
        self,
        pw: pd.DataFrame,
        pr: pd.DataFrame,
        bw: pd.DataFrame,
        br: pd.DataFrame
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Align attribution data"""
        common_dates = pw.index.intersection(pr.index).intersection(
            bw.index).intersection(br.index)
        common_assets = pw.columns.intersection(pr.columns).intersection(
            bw.columns).intersection(br.columns)

        if len(common_dates) == 0 or len(common_assets) == 0:
            return None

        return (
            pw.loc[common_dates, common_assets],
            pr.loc[common_dates, common_assets],
            bw.loc[common_dates, common_assets],
            br.loc[common_dates, common_assets]
        )


class ReportGenerator:
    """
    Generate comprehensive performance reports.

    Formats:
    - Text summary
    - HTML report
    - JSON export
    """

    def __init__(
        self,
        metrics_calculator: Optional[MetricsCalculator] = None
    ):
        self.calculator = metrics_calculator or MetricsCalculator()

    def generate_text_report(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        config: Optional[Dict] = None
    ) -> str:
        """Generate text-based report"""
        returns = equity_curve.pct_change().dropna()

        perf = self.calculator.calculate_performance_metrics(returns, equity_curve)
        risk = self.calculator.calculate_risk_metrics(returns, equity_curve)
        trade_metrics = self.calculator.calculate_trade_metrics(trades)

        report = f"""
================================================================================
                         TRADING PERFORMANCE REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
-----------------
Total Return:        {perf.total_return:>12.2%}
Annualized Return:   {perf.annualized_return:>12.2%}
Sharpe Ratio:        {perf.sharpe_ratio:>12.2f}
Max Drawdown:        {perf.max_drawdown:>12.2%}

PERFORMANCE METRICS
-------------------
Cumulative Return:   {perf.cumulative_return:>12.2%}
YTD Return:          {perf.ytd_return:>12.2%}
MTD Return:          {perf.mtd_return:>12.2%}

Risk-Adjusted Returns:
  Sharpe Ratio:      {perf.sharpe_ratio:>12.2f}
  Sortino Ratio:     {perf.sortino_ratio:>12.2f}
  Calmar Ratio:      {perf.calmar_ratio:>12.2f}
  Omega Ratio:       {perf.omega_ratio:>12.2f}

RISK METRICS
------------
Volatility:          {risk.realized_volatility:>12.2%}
Max Drawdown:        {risk.max_drawdown:>12.2%}
Avg Drawdown:        {risk.avg_drawdown:>12.2%}
Current Drawdown:    {risk.current_drawdown:>12.2%}

Value at Risk:
  VaR (95%):         {risk.var_95:>12.4f}
  VaR (99%):         {risk.var_99:>12.4f}
  CVaR (95%):        {risk.cvar_95:>12.4f}

Beta to Market:      {risk.beta:>12.2f}
Correlation:         {risk.correlation_to_market:>12.2f}

TRADE STATISTICS
----------------
Total Trades:        {trade_metrics.total_trades:>12d}
Winning Trades:      {trade_metrics.winning_trades:>12d}
Losing Trades:       {trade_metrics.losing_trades:>12d}
Win Rate:            {trade_metrics.win_rate:>12.2%}

Profit Factor:       {trade_metrics.profit_factor:>12.2f}
Payoff Ratio:        {trade_metrics.payoff_ratio:>12.2f}

Average P&L:         ${trade_metrics.avg_pnl:>11,.2f}
Average Win:         ${trade_metrics.avg_win:>11,.2f}
Average Loss:        ${trade_metrics.avg_loss:>11,.2f}
Largest Win:         ${trade_metrics.largest_win:>11,.2f}
Largest Loss:        ${trade_metrics.largest_loss:>11,.2f}

Consecutive Wins:    {trade_metrics.max_consecutive_wins:>12d}
Consecutive Losses:  {trade_metrics.max_consecutive_losses:>12d}

COSTS
-----
Total Commission:    ${trade_metrics.total_commission:>11,.2f}
Total Slippage:      ${trade_metrics.total_slippage:>11,.2f}

DISTRIBUTION ANALYSIS
--------------------
Skewness:            {perf.skewness:>12.2f}
Kurtosis:            {perf.kurtosis:>12.2f}
Stability:           {perf.stability:>12.2f}
Tail Ratio:          {perf.tail_ratio:>12.2f}

================================================================================
"""
        return report

    def generate_json_report(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate JSON report"""
        returns = equity_curve.pct_change().dropna()

        perf = self.calculator.calculate_performance_metrics(returns, equity_curve)
        risk = self.calculator.calculate_risk_metrics(returns, equity_curve)
        trade_metrics = self.calculator.calculate_trade_metrics(trades)

        return {
            'generated_at': datetime.now().isoformat(),
            'config': config or {},
            'performance': perf.to_dict(),
            'risk': {
                'var_95': risk.var_95,
                'var_99': risk.var_99,
                'cvar_95': risk.cvar_95,
                'max_drawdown': risk.max_drawdown,
                'volatility': risk.realized_volatility,
                'beta': risk.beta
            },
            'trades': trade_metrics.to_dict(),
            'equity_curve': {
                'start': equity_curve.iloc[0] if len(equity_curve) > 0 else 0,
                'end': equity_curve.iloc[-1] if len(equity_curve) > 0 else 0,
                'high': equity_curve.max(),
                'low': equity_curve.min()
            }
        }

    def generate_html_report(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        config: Optional[Dict] = None
    ) -> str:
        """Generate HTML report with charts"""
        json_data = self.generate_json_report(equity_curve, trades, config)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Trading Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #1a1a2e; color: white; padding: 20px; }}
        .metric-box {{ display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ color: #666; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: right; }}
        th {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading Performance Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <h2>Performance Summary</h2>
    <div class="metric-box">
        <div class="metric-value {'positive' if json_data['performance']['total_return'] > 0 else 'negative'}">
            {json_data['performance']['total_return']:.2%}
        </div>
        <div class="metric-label">Total Return</div>
    </div>
    <div class="metric-box">
        <div class="metric-value">{json_data['performance']['sharpe_ratio']:.2f}</div>
        <div class="metric-label">Sharpe Ratio</div>
    </div>
    <div class="metric-box">
        <div class="metric-value negative">{json_data['performance']['max_drawdown']:.2%}</div>
        <div class="metric-label">Max Drawdown</div>
    </div>
    <div class="metric-box">
        <div class="metric-value">{json_data['trades']['win_rate']:.1%}</div>
        <div class="metric-label">Win Rate</div>
    </div>

    <h2>Detailed Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Annualized Return</td><td>{json_data['performance']['annualized_return']:.2%}</td></tr>
        <tr><td>Volatility</td><td>{json_data['risk']['volatility']:.2%}</td></tr>
        <tr><td>Sortino Ratio</td><td>{json_data['performance']['sortino_ratio']:.2f}</td></tr>
        <tr><td>Calmar Ratio</td><td>{json_data['performance']['calmar_ratio']:.2f}</td></tr>
        <tr><td>VaR (95%)</td><td>{json_data['risk']['var_95']:.4f}</td></tr>
        <tr><td>CVaR (95%)</td><td>{json_data['risk']['cvar_95']:.4f}</td></tr>
    </table>

    <h2>Trade Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Trades</td><td>{json_data['trades']['total_trades']}</td></tr>
        <tr><td>Win Rate</td><td>{json_data['trades']['win_rate']:.2%}</td></tr>
        <tr><td>Profit Factor</td><td>{json_data['trades']['profit_factor']:.2f}</td></tr>
        <tr><td>Average Win</td><td>${json_data['trades']['avg_win']:,.2f}</td></tr>
        <tr><td>Average Loss</td><td>${json_data['trades']['avg_loss']:,.2f}</td></tr>
    </table>
</body>
</html>
"""
        return html


class SharpeRatioStatistics:
    """
    Advanced Sharpe Ratio Statistics.

    Based on "Advances in Financial Machine Learning" by Marcos Lopez de Prado.

    Standard Sharpe Ratio issues:
    - Susceptible to p-hacking (selection bias from multiple trials)
    - Ignores non-normality (skewness, kurtosis)
    - No statistical significance measure

    Solutions implemented:
    1. Probabilistic Sharpe Ratio (PSR): Probability that SR > benchmark
    2. Deflated Sharpe Ratio (DSR): Adjusts for multiple testing
    3. Minimum Track Record Length (minTRL): Required data for significance
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ):
        """
        Initialize SharpeRatioStatistics.

        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year (252 for daily)
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        annualize: bool = True
    ) -> float:
        """
        Calculate standard Sharpe Ratio.

        Args:
            returns: Returns series
            annualize: Whether to annualize

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        sr = excess_returns.mean() / returns.std()

        if annualize:
            sr = sr * np.sqrt(self.periods_per_year)

        return sr

    def probabilistic_sharpe_ratio(
        self,
        returns: pd.Series,
        sr_benchmark: float = 0.0
    ) -> float:
        """
        Calculate Probabilistic Sharpe Ratio (PSR).

        PSR gives the probability that the estimated Sharpe Ratio (SR*)
        exceeds a benchmark Sharpe Ratio (SR_benchmark).

        PSR = Phi((SR* - SR_benchmark) / SE(SR*))

        Where SE(SR*) accounts for non-normality through skewness and kurtosis.

        Based on AFML Chapter 14.

        Args:
            returns: Returns series
            sr_benchmark: Benchmark Sharpe ratio to exceed

        Returns:
            PSR (probability between 0 and 1)
        """
        n = len(returns)

        if n < 3 or returns.std() == 0:
            return 0.5  # Indeterminate

        # Calculate estimated Sharpe ratio (non-annualized)
        sr_star = self.calculate_sharpe_ratio(returns, annualize=False)

        # Calculate skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()  # Excess kurtosis

        # Standard error of SR accounting for non-normality
        # SE(SR*) = sqrt((1 + 0.5*SR*^2 - skew*SR* + (kurt-3)/4*SR*^2) / (n-1))
        sr_squared = sr_star ** 2
        var_sr = (1 + 0.5 * sr_squared - skewness * sr_star +
                  (kurtosis) / 4 * sr_squared) / (n - 1)

        if var_sr < 0:
            var_sr = 1 / (n - 1)  # Fallback to standard SE

        se_sr = np.sqrt(var_sr)

        if se_sr == 0:
            return 0.5

        # PSR = Phi((SR* - SR_benchmark) / SE(SR*))
        z_score = (sr_star - sr_benchmark) / se_sr
        psr = stats.norm.cdf(z_score)

        return psr

    def deflated_sharpe_ratio(
        self,
        returns: pd.Series,
        n_trials: int = 1,
        var_sharpe_trials: float = None,
        sr_benchmark: float = 0.0
    ) -> float:
        """
        Calculate Deflated Sharpe Ratio (DSR).

        DSR adjusts for multiple testing by computing the expected
        maximum Sharpe ratio under the null hypothesis (random strategy).

        This combats selection bias / p-hacking from running many backtests.

        DSR = PSR(SR*, E[max(SR)] under null)

        Based on AFML Chapter 14.

        Args:
            returns: Returns series
            n_trials: Number of independent trials (backtests) run
            var_sharpe_trials: Variance of SR across trials (estimated if None)
            sr_benchmark: Minimum acceptable SR (default: 0)

        Returns:
            DSR (probability between 0 and 1)
        """
        n = len(returns)

        if n < 3 or returns.std() == 0:
            return 0.5

        # Calculate estimated Sharpe ratio
        sr_star = self.calculate_sharpe_ratio(returns, annualize=False)

        # Estimate variance of SR across trials if not provided
        if var_sharpe_trials is None:
            # Default: assume standard SE
            var_sharpe_trials = 1 / (n - 1)

        # Expected maximum SR under null hypothesis
        # E[max(SR)] = sqrt(V[SR]) * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e)))
        # Approximation using Euler-Mascheroni constant
        gamma = 0.5772156649  # Euler-Mascheroni constant

        if n_trials > 1:
            z1 = stats.norm.ppf(1 - 1 / n_trials)
            z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
            e_max_sr = np.sqrt(var_sharpe_trials) * ((1 - gamma) * z1 + gamma * z2)
        else:
            e_max_sr = 0

        # Adjusted benchmark
        sr_benchmark_adj = max(sr_benchmark, e_max_sr)

        # Calculate PSR with adjusted benchmark
        dsr = self.probabilistic_sharpe_ratio(returns, sr_benchmark_adj)

        return dsr

    def minimum_track_record_length(
        self,
        returns: pd.Series,
        sr_benchmark: float = 0.0,
        confidence: float = 0.95
    ) -> int:
        """
        Calculate Minimum Track Record Length (minTRL).

        minTRL is the minimum number of observations needed to reject
        the null hypothesis that the true SR <= SR_benchmark.

        Based on AFML Chapter 14.

        Args:
            returns: Returns series
            sr_benchmark: Benchmark SR to exceed
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Minimum track record length (number of observations)
        """
        n = len(returns)

        if n < 3 or returns.std() == 0:
            return float('inf')

        # Calculate estimated Sharpe ratio (non-annualized)
        sr_star = self.calculate_sharpe_ratio(returns, annualize=False)

        if sr_star <= sr_benchmark:
            return float('inf')  # Cannot reject null

        # Calculate skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # z-score for confidence level
        z_alpha = stats.norm.ppf(confidence)

        # minTRL formula (from AFML)
        # minTRL = 1 + (1 + 0.5*SR*^2 - skew*SR* + (kurt-3)/4*SR*^2) * (z_alpha / (SR* - SR_b))^2
        sr_squared = sr_star ** 2
        variance_factor = 1 + 0.5 * sr_squared - skewness * sr_star + (kurtosis) / 4 * sr_squared
        z_ratio = z_alpha / (sr_star - sr_benchmark)

        min_trl = 1 + variance_factor * (z_ratio ** 2)

        return int(np.ceil(min_trl))

    def sharpe_ratio_confidence_interval(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for Sharpe Ratio.

        Args:
            returns: Returns series
            confidence: Confidence level

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(returns)

        if n < 3 or returns.std() == 0:
            return (0.0, 0.0)

        sr_star = self.calculate_sharpe_ratio(returns, annualize=False)

        # Calculate skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Standard error
        sr_squared = sr_star ** 2
        var_sr = (1 + 0.5 * sr_squared - skewness * sr_star +
                  (kurtosis) / 4 * sr_squared) / (n - 1)

        if var_sr < 0:
            var_sr = 1 / (n - 1)

        se_sr = np.sqrt(var_sr)

        # Confidence interval
        z = stats.norm.ppf((1 + confidence) / 2)

        lower = sr_star - z * se_sr
        upper = sr_star + z * se_sr

        # Annualize
        annualization_factor = np.sqrt(self.periods_per_year)
        lower *= annualization_factor
        upper *= annualization_factor

        return (lower, upper)

    def generate_sharpe_report(
        self,
        returns: pd.Series,
        n_trials: int = 1,
        sr_benchmark: float = 0.0,
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Generate comprehensive Sharpe ratio analysis report.

        Args:
            returns: Returns series
            n_trials: Number of backtests/trials run
            sr_benchmark: Benchmark SR
            confidence: Confidence level

        Returns:
            Dictionary with comprehensive Sharpe statistics
        """
        sr = self.calculate_sharpe_ratio(returns, annualize=True)
        psr = self.probabilistic_sharpe_ratio(returns, sr_benchmark)
        dsr = self.deflated_sharpe_ratio(returns, n_trials, sr_benchmark=sr_benchmark)
        min_trl = self.minimum_track_record_length(returns, sr_benchmark, confidence)
        ci = self.sharpe_ratio_confidence_interval(returns, confidence)

        # Additional statistics
        n = len(returns)
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Interpretation
        is_significant = psr > confidence
        has_enough_data = n >= min_trl
        is_deflated_significant = dsr > confidence

        return {
            'sharpe_ratio': sr,
            'sharpe_ratio_annualized': sr,
            'sharpe_ratio_daily': sr / np.sqrt(self.periods_per_year),
            'probabilistic_sr': psr,
            'deflated_sr': dsr,
            'minimum_track_record': min_trl,
            'confidence_interval': ci,
            'n_observations': n,
            'n_trials': n_trials,
            'sr_benchmark': sr_benchmark,
            'confidence_level': confidence,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_significant': is_significant,
            'has_enough_data': has_enough_data,
            'is_deflated_significant': is_deflated_significant,
            'interpretation': self._interpret_results(
                sr, psr, dsr, min_trl, n, is_significant,
                has_enough_data, is_deflated_significant
            )
        }

    def _interpret_results(
        self,
        sr: float,
        psr: float,
        dsr: float,
        min_trl: int,
        n: int,
        is_significant: bool,
        has_enough_data: bool,
        is_deflated_significant: bool
    ) -> str:
        """Generate human-readable interpretation."""
        interpretation = []

        # SR interpretation
        if sr > 2.0:
            interpretation.append("Exceptional Sharpe ratio (>2.0)")
        elif sr > 1.0:
            interpretation.append("Strong Sharpe ratio (>1.0)")
        elif sr > 0.5:
            interpretation.append("Acceptable Sharpe ratio (>0.5)")
        elif sr > 0:
            interpretation.append("Weak positive Sharpe ratio")
        else:
            interpretation.append("Negative Sharpe ratio - strategy underperforms")

        # PSR interpretation
        if is_significant:
            interpretation.append(f"PSR significant ({psr:.1%} > 95%)")
        else:
            interpretation.append(f"PSR not significant ({psr:.1%} < 95%)")

        # Track record
        if has_enough_data:
            interpretation.append(f"Sufficient data ({n} >= {min_trl} minTRL)")
        else:
            interpretation.append(f"Need more data ({n} < {min_trl} minTRL)")

        # DSR interpretation
        if is_deflated_significant:
            interpretation.append("Survives multiple testing adjustment (DSR)")
        else:
            interpretation.append("May be result of selection bias (low DSR)")

        return " | ".join(interpretation)


def calculate_psr(
    returns: pd.Series,
    sr_benchmark: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Convenience function for Probabilistic Sharpe Ratio.

    Args:
        returns: Returns series
        sr_benchmark: Benchmark SR to exceed
        periods_per_year: Periods per year

    Returns:
        PSR value
    """
    calc = SharpeRatioStatistics(periods_per_year=periods_per_year)
    return calc.probabilistic_sharpe_ratio(returns, sr_benchmark)


def calculate_dsr(
    returns: pd.Series,
    n_trials: int,
    sr_benchmark: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Convenience function for Deflated Sharpe Ratio.

    Args:
        returns: Returns series
        n_trials: Number of trials/backtests
        sr_benchmark: Benchmark SR
        periods_per_year: Periods per year

    Returns:
        DSR value
    """
    calc = SharpeRatioStatistics(periods_per_year=periods_per_year)
    return calc.deflated_sharpe_ratio(returns, n_trials, sr_benchmark=sr_benchmark)


class DeflatedSharpeRatio:
    """
    Institutional-Grade Deflated Sharpe Ratio Implementation.

    Implements the complete DSR framework from Marcos Lopez de Prado's
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
    Overfitting, and the Haircut Sharpe Ratio".

    This class addresses the critical problem of selection bias in quantitative
    finance, where researchers run many backtests and report only the best results.

    Key Features:
    1. Automatic estimation of independent trials from strategy variants
    2. Proper adjustment for non-normality (skewness, kurtosis)
    3. Haircut Sharpe Ratio calculation
    4. Overfitting probability estimation
    5. Integration with backtest reporting

    The formula for DSR:
    DSR = PSR(SR*, E[max(SR)]) where E[max(SR)] is the expected maximum
    Sharpe ratio under the null hypothesis (random strategy).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ):
        """
        Initialize DeflatedSharpeRatio calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 0)
            periods_per_year: Trading periods per year (252 for daily)
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self._trials_history: List[Dict] = []

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        annualize: bool = True
    ) -> float:
        """
        Calculate standard Sharpe Ratio.

        SR = (E[R] - Rf) / std(R)

        Args:
            returns: Returns series
            annualize: Whether to annualize (default: True)

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        sr = excess_returns.mean() / returns.std()

        if annualize:
            sr *= np.sqrt(self.periods_per_year)

        return sr

    def calculate_sharpe_ratio_standard_error(
        self,
        returns: pd.Series
    ) -> float:
        """
        Calculate standard error of Sharpe Ratio accounting for non-normality.

        SE(SR*) = sqrt((1 - gamma3*SR* + (gamma4-1)/4*SR*^2) / (T-1))

        Where:
        - gamma3 = skewness
        - gamma4 = kurtosis (excess)
        - T = number of observations

        This formula accounts for the fact that returns are typically
        non-normal (fat tails, negative skew).

        Args:
            returns: Returns series

        Returns:
            Standard error of Sharpe ratio
        """
        n = len(returns)
        if n < 4:
            return float('inf')

        sr = self.calculate_sharpe_ratio(returns, annualize=False)
        skew = returns.skew()
        kurt = returns.kurtosis()  # Excess kurtosis

        # SE formula from Bailey & Lopez de Prado
        sr_squared = sr ** 2
        variance = (1 - skew * sr + (kurt / 4) * sr_squared) / (n - 1)

        if variance < 0:
            # Fallback to standard SE if formula gives negative variance
            variance = 1 / (n - 1)

        return np.sqrt(variance)

    def probabilistic_sharpe_ratio(
        self,
        returns: pd.Series,
        sr_benchmark: float = 0.0
    ) -> float:
        """
        Calculate Probabilistic Sharpe Ratio (PSR).

        PSR = Phi((SR* - SR_benchmark) / SE(SR*))

        PSR gives the probability that the true Sharpe Ratio exceeds
        the benchmark, accounting for estimation error and non-normality.

        Args:
            returns: Returns series
            sr_benchmark: Benchmark SR to compare against

        Returns:
            PSR (probability between 0 and 1)
        """
        n = len(returns)
        if n < 4:
            return 0.5

        sr = self.calculate_sharpe_ratio(returns, annualize=False)
        se = self.calculate_sharpe_ratio_standard_error(returns)

        if se <= 0 or se == float('inf'):
            return 0.5

        z_score = (sr - sr_benchmark) / se
        return stats.norm.cdf(z_score)

    def estimate_independent_trials(
        self,
        strategy_returns: List[pd.Series],
        correlation_threshold: float = 0.5
    ) -> int:
        """
        Estimate the number of independent trials from correlated strategies.

        When researchers test multiple strategy variants, these are often
        highly correlated. This method estimates the effective number of
        independent trials using the eigenvalue decomposition approach.

        N_independent = T / (1 + (N-1) * avg_correlation)

        Or using eigenvalue method:
        N_independent = (sum(eigenvalues))^2 / sum(eigenvalues^2)

        Args:
            strategy_returns: List of return series for each strategy variant
            correlation_threshold: Threshold for considering strategies independent

        Returns:
            Estimated number of independent trials
        """
        if len(strategy_returns) <= 1:
            return len(strategy_returns)

        # Build correlation matrix of strategy returns
        n_strategies = len(strategy_returns)

        # Align returns to common dates
        combined = pd.concat(
            [r.rename(f'strat_{i}') for i, r in enumerate(strategy_returns)],
            axis=1
        ).dropna()

        if len(combined) < 10:
            return n_strategies

        corr_matrix = combined.corr().values

        # Method 1: Eigenvalue-based (more accurate)
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]

        if len(eigenvalues) > 0:
            n_independent_eigen = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        else:
            n_independent_eigen = n_strategies

        # Method 2: Average correlation
        # Get upper triangle (excluding diagonal)
        upper_tri = corr_matrix[np.triu_indices(n_strategies, k=1)]
        if len(upper_tri) > 0:
            avg_corr = np.mean(np.abs(upper_tri))
            n_independent_corr = n_strategies / (1 + (n_strategies - 1) * avg_corr)
        else:
            n_independent_corr = n_strategies

        # Use the more conservative estimate
        n_independent = min(n_independent_eigen, n_independent_corr)

        return max(1, int(np.ceil(n_independent)))

    def estimate_trials_from_variance(
        self,
        returns: pd.Series,
        strategy_sharpe_ratios: Optional[List[float]] = None
    ) -> int:
        """
        Estimate number of trials from the variance of Sharpe ratios.

        If multiple strategy variants were tested, the variance of their
        Sharpe ratios can be used to infer the number of independent trials.

        Formula: N = (SR_max - SR_mean)^2 / Var(SR) * correction_factor

        Args:
            returns: Primary strategy returns
            strategy_sharpe_ratios: List of SRs from all tested variants

        Returns:
            Estimated number of independent trials
        """
        if strategy_sharpe_ratios is None or len(strategy_sharpe_ratios) < 2:
            # Default estimate based on typical research practices
            # Assume 50-100 variants tested
            return 50

        sr_array = np.array(strategy_sharpe_ratios)
        sr_max = sr_array.max()
        sr_mean = sr_array.mean()
        sr_var = sr_array.var()

        if sr_var <= 0:
            return len(strategy_sharpe_ratios)

        # Bailey & Lopez de Prado formula
        # Expected max of N iid draws from standard normal
        n_reported = len(strategy_sharpe_ratios)
        expected_max_factor = np.sqrt(2 * np.log(n_reported))

        n_estimated = max(1, int(((sr_max - sr_mean) / np.sqrt(sr_var)) ** 2 / expected_max_factor))

        return n_estimated

    def expected_max_sharpe_ratio(
        self,
        n_trials: int,
        variance_sharpe: float = None,
        n_observations: int = 252
    ) -> float:
        """
        Calculate expected maximum Sharpe Ratio under the null hypothesis.

        Under the null (random strategy), the expected maximum SR from
        N independent trials is:

        E[max(SR)] = sqrt(V(SR)) * [(1-gamma)*Phi^-1(1-1/N) + gamma*Phi^-1(1-1/(N*e))]

        Where gamma is the Euler-Mascheroni constant (0.5772...)

        Args:
            n_trials: Number of independent trials
            variance_sharpe: Variance of SR estimate (default: 1/(T-1))
            n_observations: Number of return observations

        Returns:
            Expected maximum Sharpe ratio under null
        """
        if n_trials <= 1:
            return 0.0

        gamma = 0.5772156649  # Euler-Mascheroni constant

        # Variance of SR under standard assumptions
        if variance_sharpe is None:
            variance_sharpe = 1.0 / (n_observations - 1)

        std_sharpe = np.sqrt(variance_sharpe)

        # Expected maximum of N iid standard normals
        z1 = stats.norm.ppf(1 - 1 / n_trials)
        z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))

        e_max = std_sharpe * ((1 - gamma) * z1 + gamma * z2)

        return e_max

    def deflated_sharpe_ratio(
        self,
        returns: pd.Series,
        n_trials: int = None,
        strategy_returns_list: Optional[List[pd.Series]] = None,
        sr_benchmark: float = 0.0
    ) -> float:
        """
        Calculate the Deflated Sharpe Ratio.

        DSR adjusts for selection bias from multiple testing by computing
        the probability that the observed SR exceeds the expected maximum
        SR under the null hypothesis.

        DSR = PSR(SR*, max(SR_benchmark, E[max(SR)]))

        A DSR < 0.95 indicates the strategy may be a false positive
        (result of data mining rather than genuine alpha).

        Args:
            returns: Strategy returns
            n_trials: Number of independent trials (auto-estimated if None)
            strategy_returns_list: Returns from all tested variants
            sr_benchmark: Minimum acceptable SR

        Returns:
            DSR (probability between 0 and 1)
        """
        n = len(returns)
        if n < 4:
            return 0.5

        # Calculate observed SR
        sr_observed = self.calculate_sharpe_ratio(returns, annualize=False)

        # Estimate number of trials if not provided
        if n_trials is None:
            if strategy_returns_list is not None:
                n_trials = self.estimate_independent_trials(strategy_returns_list)
            else:
                # Default assumption: moderate selection bias
                n_trials = 50

        # Calculate variance of SR
        skew = returns.skew()
        kurt = returns.kurtosis()
        sr_squared = sr_observed ** 2
        var_sr = (1 - skew * sr_observed + (kurt / 4) * sr_squared) / (n - 1)
        var_sr = max(var_sr, 1 / (n - 1))

        # Expected maximum SR under null
        e_max_sr = self.expected_max_sharpe_ratio(n_trials, var_sr, n)

        # Adjusted benchmark
        sr_benchmark_adj = max(sr_benchmark, e_max_sr)

        # Calculate DSR = PSR with adjusted benchmark
        dsr = self.probabilistic_sharpe_ratio(returns, sr_benchmark_adj)

        return dsr

    def haircut_sharpe_ratio(
        self,
        returns: pd.Series,
        n_trials: int = None,
        strategy_returns_list: Optional[List[pd.Series]] = None
    ) -> float:
        """
        Calculate the Haircut Sharpe Ratio.

        The Haircut SR is the SR adjusted downward for expected multiple
        testing bias. It represents what the "true" SR might be after
        accounting for selection effects.

        Haircut_SR = SR_observed - E[max(SR)] * (1 - DSR)

        Args:
            returns: Strategy returns
            n_trials: Number of independent trials
            strategy_returns_list: Returns from all tested variants

        Returns:
            Haircut-adjusted Sharpe ratio
        """
        sr_observed = self.calculate_sharpe_ratio(returns, annualize=True)
        dsr = self.deflated_sharpe_ratio(returns, n_trials, strategy_returns_list)

        n = len(returns)

        # Estimate variance and expected max
        if n_trials is None:
            n_trials = 50

        var_sr = 1 / (n - 1)
        e_max_sr = self.expected_max_sharpe_ratio(n_trials, var_sr, n)

        # Annualize the expected max
        e_max_sr_ann = e_max_sr * np.sqrt(self.periods_per_year)

        # Haircut based on DSR
        haircut = e_max_sr_ann * (1 - dsr)
        haircut_sr = sr_observed - haircut

        return max(0, haircut_sr)

    def probability_of_overfitting(
        self,
        in_sample_returns: pd.Series,
        out_of_sample_returns: pd.Series,
        n_trials: int = None
    ) -> float:
        """
        Calculate the probability that a strategy is overfit.

        Uses the ratio of out-of-sample to in-sample Sharpe ratios
        compared to what would be expected from a non-overfit strategy.

        Args:
            in_sample_returns: In-sample backtest returns
            out_of_sample_returns: Out-of-sample returns
            n_trials: Number of trials tested

        Returns:
            Probability of overfitting (0-1)
        """
        sr_is = self.calculate_sharpe_ratio(in_sample_returns, annualize=True)
        sr_oos = self.calculate_sharpe_ratio(out_of_sample_returns, annualize=True)

        if sr_is <= 0:
            return 1.0

        # Ratio of OOS to IS Sharpe
        sr_ratio = sr_oos / sr_is

        # Expected ratio under no overfitting (theoretical: ~1)
        # Under overfitting, ratio is typically < 0.5
        if sr_ratio >= 1.0:
            prob_overfit = 0.0
        elif sr_ratio <= 0:
            prob_overfit = 1.0
        else:
            # Use DSR to estimate overfitting probability
            dsr = self.deflated_sharpe_ratio(in_sample_returns, n_trials)
            prob_overfit = 1 - (sr_ratio * dsr)

        return max(0, min(1, prob_overfit))

    def generate_report(
        self,
        returns: pd.Series,
        n_trials: int = None,
        strategy_returns_list: Optional[List[pd.Series]] = None,
        strategy_name: str = "Strategy"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive DSR analysis report.

        Args:
            returns: Strategy returns
            n_trials: Number of trials (auto-estimated if None)
            strategy_returns_list: Returns from all tested variants
            strategy_name: Name for reporting

        Returns:
            Dictionary with full DSR analysis
        """
        n = len(returns)
        sr = self.calculate_sharpe_ratio(returns, annualize=True)
        sr_se = self.calculate_sharpe_ratio_standard_error(returns)
        psr = self.probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
        dsr = self.deflated_sharpe_ratio(returns, n_trials, strategy_returns_list)
        haircut_sr = self.haircut_sharpe_ratio(returns, n_trials, strategy_returns_list)

        # Confidence interval
        z = stats.norm.ppf(0.975)  # 95% CI
        sr_daily = self.calculate_sharpe_ratio(returns, annualize=False)
        ci_lower = (sr_daily - z * sr_se) * np.sqrt(self.periods_per_year)
        ci_upper = (sr_daily + z * sr_se) * np.sqrt(self.periods_per_year)

        # Interpretation
        if dsr >= 0.95:
            significance = "SIGNIFICANT"
            interpretation = "Strategy passes multiple testing adjustment"
        elif dsr >= 0.80:
            significance = "MARGINAL"
            interpretation = "Strategy shows some evidence but needs more data"
        else:
            significance = "INSIGNIFICANT/OVERFIT"
            interpretation = "Strategy likely result of data mining"

        return {
            'strategy_name': strategy_name,
            'n_observations': n,
            'n_trials_estimated': n_trials or 50,

            # Sharpe Ratios
            'sharpe_ratio': sr,
            'sharpe_ratio_se': sr_se * np.sqrt(self.periods_per_year),
            'sharpe_ratio_ci_95': (ci_lower, ci_upper),

            # Probabilistic metrics
            'probabilistic_sr': psr,
            'deflated_sr': dsr,
            'haircut_sr': haircut_sr,

            # Distribution
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),

            # Interpretation
            'significance': significance,
            'interpretation': interpretation,
            'passes_95_threshold': dsr >= 0.95,

            # Expected values under null
            'expected_max_sr_null': self.expected_max_sharpe_ratio(
                n_trials or 50, 1 / (n - 1), n
            ) * np.sqrt(self.periods_per_year)
        }


# Updated convenience function with better trials estimation
def calculate_dsr_with_trials_estimation(
    returns: pd.Series,
    strategy_variants: Optional[List[pd.Series]] = None,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate DSR with automatic trials estimation.

    This is the recommended function for institutional use as it
    properly estimates the number of independent trials from
    correlated strategy variants.

    Args:
        returns: Primary strategy returns
        strategy_variants: List of returns from all tested strategy variants
        periods_per_year: Trading periods per year

    Returns:
        Dictionary with DSR, haircut SR, and diagnostics
    """
    calculator = DeflatedSharpeRatio(periods_per_year=periods_per_year)

    if strategy_variants:
        n_trials = calculator.estimate_independent_trials(strategy_variants)
    else:
        n_trials = None  # Will use default

    report = calculator.generate_report(
        returns,
        n_trials=n_trials,
        strategy_returns_list=strategy_variants
    )

    return report
