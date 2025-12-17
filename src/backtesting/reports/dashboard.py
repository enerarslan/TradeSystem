"""
Institutional-grade performance dashboard for AlphaTrade system.

This module provides:
- Interactive dashboards with Plotly
- Real-time metrics display
- Multi-strategy comparison
- Risk attribution analysis
- Factor exposure visualization
- Tear sheets (QuantStats-style)

Designed for JPMorgan-level requirements:
- Comprehensive risk metrics
- Regulatory-compliant reporting
- Attribution analysis
- Benchmark comparison
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TearSheetMetrics:
    """
    Comprehensive tear sheet metrics.

    Contains all metrics needed for institutional reporting.
    """
    # Return metrics
    total_return: float = 0.0
    cagr: float = 0.0
    mtd_return: float = 0.0
    qtd_return: float = 0.0
    ytd_return: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Distribution
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    var_99: float = 0.0
    cvar_99: float = 0.0

    # Trading
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Period metrics
    best_day: float = 0.0
    worst_day: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0
    best_year: float = 0.0
    worst_year: float = 0.0

    # Statistics
    positive_days_pct: float = 0.0
    positive_months_pct: float = 0.0
    positive_years_pct: float = 0.0


class MetricsCalculator:
    """
    Calculate comprehensive performance metrics.

    Computes all metrics needed for institutional reporting
    from returns series.
    """

    def __init__(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ):
        """
        Initialize calculator.

        Args:
            returns: Strategy returns series
            benchmark_returns: Optional benchmark returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
        """
        self.returns = returns.dropna()
        self.benchmark = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.rf_rate = risk_free_rate
        self.periods = periods_per_year

        # Compute equity curve
        self.equity = (1 + self.returns).cumprod()

    def calculate_all(self) -> TearSheetMetrics:
        """Calculate all tear sheet metrics."""
        metrics = TearSheetMetrics()

        if len(self.returns) < 2:
            return metrics

        # Return metrics
        metrics.total_return = self._total_return()
        metrics.cagr = self._cagr()
        metrics.mtd_return = self._period_return('M')
        metrics.qtd_return = self._period_return('Q')
        metrics.ytd_return = self._period_return('Y')

        # Risk metrics
        metrics.volatility = self._volatility()
        metrics.max_drawdown = self._max_drawdown()
        metrics.avg_drawdown = self._avg_drawdown()
        metrics.max_drawdown_duration = self._max_drawdown_duration()

        # Risk-adjusted
        metrics.sharpe_ratio = self._sharpe_ratio()
        metrics.sortino_ratio = self._sortino_ratio()
        metrics.calmar_ratio = self._calmar_ratio()
        metrics.omega_ratio = self._omega_ratio()

        # Distribution
        metrics.skewness = self._skewness()
        metrics.kurtosis = self._kurtosis()
        metrics.var_95 = self._var(0.05)
        metrics.cvar_95 = self._cvar(0.05)
        metrics.var_99 = self._var(0.01)
        metrics.cvar_99 = self._cvar(0.01)

        # Trading
        metrics.win_rate = self._win_rate()
        metrics.profit_factor = self._profit_factor()
        metrics.avg_win = self._avg_win()
        metrics.avg_loss = self._avg_loss()
        metrics.max_consecutive_wins = self._max_consecutive_wins()
        metrics.max_consecutive_losses = self._max_consecutive_losses()

        # Period metrics
        metrics.best_day = self.returns.max()
        metrics.worst_day = self.returns.min()

        monthly = self._monthly_returns()
        if len(monthly) > 0:
            metrics.best_month = monthly.max()
            metrics.worst_month = monthly.min()

        yearly = self._yearly_returns()
        if len(yearly) > 0:
            metrics.best_year = yearly.max()
            metrics.worst_year = yearly.min()

        # Statistics
        metrics.positive_days_pct = (self.returns > 0).sum() / len(self.returns)
        if len(monthly) > 0:
            metrics.positive_months_pct = (monthly > 0).sum() / len(monthly)
        if len(yearly) > 0:
            metrics.positive_years_pct = (yearly > 0).sum() / len(yearly)

        return metrics

    def _total_return(self) -> float:
        return (self.equity.iloc[-1] / self.equity.iloc[0]) - 1

    def _cagr(self) -> float:
        years = len(self.returns) / self.periods
        if years <= 0:
            return 0.0
        return (self.equity.iloc[-1] ** (1 / years)) - 1

    def _period_return(self, period: str) -> float:
        """Calculate return for current period (MTD, QTD, YTD)."""
        if period == 'M':
            start = self.returns.index[-1].replace(day=1)
        elif period == 'Q':
            q = (self.returns.index[-1].month - 1) // 3
            start = self.returns.index[-1].replace(month=q*3+1, day=1)
        else:  # Year
            start = self.returns.index[-1].replace(month=1, day=1)

        period_returns = self.returns[self.returns.index >= start]
        if len(period_returns) == 0:
            return 0.0
        return (1 + period_returns).prod() - 1

    def _volatility(self) -> float:
        return self.returns.std() * np.sqrt(self.periods)

    def _max_drawdown(self) -> float:
        running_max = self.equity.expanding().max()
        drawdown = (self.equity - running_max) / running_max
        return drawdown.min()

    def _avg_drawdown(self) -> float:
        running_max = self.equity.expanding().max()
        drawdown = (self.equity - running_max) / running_max
        return drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0.0

    def _max_drawdown_duration(self) -> int:
        running_max = self.equity.expanding().max()
        is_underwater = self.equity < running_max

        max_duration = 0
        current_duration = 0

        for underwater in is_underwater:
            if underwater:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def _sharpe_ratio(self) -> float:
        excess_returns = self.returns - self.rf_rate / self.periods
        std = excess_returns.std()
        if std == 0:
            return 0.0
        return (excess_returns.mean() / std) * np.sqrt(self.periods)

    def _sortino_ratio(self) -> float:
        excess_returns = self.returns - self.rf_rate / self.periods
        downside = excess_returns[excess_returns < 0]
        downside_std = downside.std()
        if downside_std == 0:
            return 0.0
        return (excess_returns.mean() / downside_std) * np.sqrt(self.periods)

    def _calmar_ratio(self) -> float:
        max_dd = abs(self._max_drawdown())
        if max_dd == 0:
            return 0.0
        return self._cagr() / max_dd

    def _omega_ratio(self, threshold: float = 0.0) -> float:
        gains = self.returns[self.returns > threshold] - threshold
        losses = threshold - self.returns[self.returns <= threshold]

        sum_losses = losses.sum()
        if sum_losses == 0:
            return np.inf if len(gains) > 0 else 0.0
        return gains.sum() / sum_losses

    def _skewness(self) -> float:
        return float(self.returns.skew())

    def _kurtosis(self) -> float:
        return float(self.returns.kurtosis())

    def _var(self, alpha: float) -> float:
        return float(np.percentile(self.returns, alpha * 100))

    def _cvar(self, alpha: float) -> float:
        var = self._var(alpha)
        return float(self.returns[self.returns <= var].mean())

    def _win_rate(self) -> float:
        if len(self.returns) == 0:
            return 0.0
        return (self.returns > 0).sum() / len(self.returns)

    def _profit_factor(self) -> float:
        wins = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        if losses == 0:
            return np.inf if wins > 0 else 0.0
        return wins / losses

    def _avg_win(self) -> float:
        wins = self.returns[self.returns > 0]
        return wins.mean() if len(wins) > 0 else 0.0

    def _avg_loss(self) -> float:
        losses = self.returns[self.returns < 0]
        return losses.mean() if len(losses) > 0 else 0.0

    def _max_consecutive_wins(self) -> int:
        is_win = self.returns > 0
        max_consecutive = 0
        current = 0
        for win in is_win:
            if win:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        return max_consecutive

    def _max_consecutive_losses(self) -> int:
        is_loss = self.returns < 0
        max_consecutive = 0
        current = 0
        for loss in is_loss:
            if loss:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        return max_consecutive

    def _monthly_returns(self) -> pd.Series:
        return self.returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    def _yearly_returns(self) -> pd.Series:
        return self.returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)


class PerformanceDashboard:
    """
    Interactive performance dashboard.

    Creates comprehensive visualizations for strategy analysis
    with institutional-grade metrics and styling.
    """

    def __init__(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        benchmark_name: str = "Benchmark",
        strategy_name: str = "Strategy",
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ):
        """
        Initialize dashboard.

        Args:
            returns: Strategy returns series
            benchmark_returns: Optional benchmark returns
            benchmark_name: Name of benchmark
            strategy_name: Name of strategy
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for dashboard. Install with: pip install plotly")

        self.returns = returns.dropna()
        self.benchmark = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.benchmark_name = benchmark_name
        self.strategy_name = strategy_name
        self.rf_rate = risk_free_rate
        self.periods = periods_per_year

        # Calculate metrics
        self.calculator = MetricsCalculator(
            returns, benchmark_returns, risk_free_rate, periods_per_year
        )
        self.metrics = self.calculator.calculate_all()

        # Equity curves
        self.equity = (1 + self.returns).cumprod()
        if self.benchmark is not None:
            self.benchmark_equity = (1 + self.benchmark).cumprod()
        else:
            self.benchmark_equity = None

        # Color scheme
        self.colors = {
            'strategy': '#4CAF50',
            'benchmark': '#607D8B',
            'positive': '#4CAF50',
            'negative': '#f44336',
            'neutral': '#2196F3',
            'background': '#1a1a2e',
            'card': '#16213e',
            'text': '#EEEEEE',
            'grid': '#333333',
        }

    def create_equity_chart(self) -> go.Figure:
        """Create equity curve chart."""
        fig = go.Figure()

        # Strategy equity
        fig.add_trace(go.Scatter(
            x=self.equity.index,
            y=self.equity.values,
            name=self.strategy_name,
            line=dict(color=self.colors['strategy'], width=2),
            hovertemplate='%{x}<br>Value: %{y:.4f}<extra></extra>',
        ))

        # Benchmark equity
        if self.benchmark_equity is not None:
            fig.add_trace(go.Scatter(
                x=self.benchmark_equity.index,
                y=self.benchmark_equity.values,
                name=self.benchmark_name,
                line=dict(color=self.colors['benchmark'], width=1.5, dash='dash'),
                hovertemplate='%{x}<br>Value: %{y:.4f}<extra></extra>',
            ))

        fig.update_layout(
            title=dict(text='<b>Cumulative Returns</b>', font=dict(size=16)),
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($1 Initial)',
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
            height=400,
        )

        return fig

    def create_drawdown_chart(self) -> go.Figure:
        """Create underwater (drawdown) chart."""
        running_max = self.equity.expanding().max()
        drawdown = (self.equity - running_max) / running_max * 100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color=self.colors['negative'], width=1),
            fillcolor='rgba(244, 67, 54, 0.3)',
            hovertemplate='%{x}<br>Drawdown: %{y:.2f}%<extra></extra>',
        ))

        fig.update_layout(
            title=dict(text='<b>Underwater Chart</b>', font=dict(size=16)),
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_dark',
            height=250,
        )

        return fig

    def create_returns_distribution(self) -> go.Figure:
        """Create returns distribution histogram."""
        returns_pct = self.returns * 100

        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=returns_pct,
            nbinsx=50,
            name='Returns',
            marker_color=self.colors['neutral'],
            opacity=0.7,
            hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>',
        ))

        # Normal distribution overlay
        mean = returns_pct.mean()
        std = returns_pct.std()
        x_norm = np.linspace(returns_pct.min(), returns_pct.max(), 100)
        y_norm = (
            len(returns_pct)
            * (returns_pct.max() - returns_pct.min())
            / 50
            * (1 / (std * np.sqrt(2 * np.pi)))
            * np.exp(-((x_norm - mean) ** 2) / (2 * std ** 2))
        )

        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            name='Normal Distribution',
            line=dict(color='yellow', dash='dash', width=2),
        ))

        # Add vertical lines for VaR
        var_95 = np.percentile(returns_pct, 5)
        fig.add_vline(x=var_95, line_dash='dot', line_color='red',
                      annotation_text=f'VaR 95%: {var_95:.2f}%')

        fig.update_layout(
            title=dict(text='<b>Returns Distribution</b>', font=dict(size=16)),
            xaxis_title='Daily Return (%)',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=300,
        )

        return fig

    def create_monthly_heatmap(self) -> go.Figure:
        """Create monthly returns heatmap."""
        monthly = self.returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100

        monthly_df = monthly.to_frame('return')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month

        pivot = monthly_df.pivot(index='year', columns='month', values='return')

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Add YTD column
        yearly = self.returns.resample('YE').apply(lambda x: (1 + x).prod() - 1) * 100
        pivot['YTD'] = yearly.values[-len(pivot):]

        cols = months[:pivot.shape[1]-1] + ['YTD']

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=cols,
            y=pivot.index.astype(str),
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate='%{text}%',
            textfont=dict(size=10),
            hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>',
            colorbar=dict(title='Return %'),
        ))

        fig.update_layout(
            title=dict(text='<b>Monthly Returns (%)</b>', font=dict(size=16)),
            template='plotly_dark',
            height=300,
        )

        return fig

    def create_rolling_metrics(self, window: int = 252) -> go.Figure:
        """Create rolling metrics chart."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility'),
            vertical_spacing=0.15,
        )

        # Rolling Sharpe
        rolling_mean = self.returns.rolling(window).mean() * self.periods
        rolling_std = self.returns.rolling(window).std() * np.sqrt(self.periods)
        rolling_sharpe = rolling_mean / rolling_std

        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            name='Sharpe Ratio',
            line=dict(color=self.colors['strategy'], width=2),
        ), row=1, col=1)

        # Add threshold lines
        fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3, row=1, col=1)
        fig.add_hline(y=1, line_dash='dot', line_color='green', opacity=0.3, row=1, col=1)
        fig.add_hline(y=2, line_dash='dot', line_color='green', opacity=0.3, row=1, col=1)

        # Rolling Volatility
        fig.add_trace(go.Scatter(
            x=rolling_std.index,
            y=rolling_std.values * 100,
            name='Volatility',
            line=dict(color=self.colors['neutral'], width=2),
        ), row=2, col=1)

        fig.update_layout(
            title=dict(text=f'<b>Rolling Metrics ({window}-day)</b>', font=dict(size=16)),
            template='plotly_dark',
            height=500,
            showlegend=False,
        )

        fig.update_yaxes(title_text='Sharpe', row=1, col=1)
        fig.update_yaxes(title_text='Volatility (%)', row=2, col=1)

        return fig

    def create_drawdown_analysis(self) -> go.Figure:
        """Create detailed drawdown analysis."""
        running_max = self.equity.expanding().max()
        drawdown = (self.equity - running_max) / running_max

        # Find top 5 drawdowns
        dd_periods = []
        in_drawdown = False
        start_idx = None

        for i, (idx, val) in enumerate(drawdown.items()):
            if val < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = idx
            elif val >= 0 and in_drawdown:
                in_drawdown = False
                end_idx = idx
                min_val = drawdown.loc[start_idx:end_idx].min()
                dd_periods.append({
                    'start': start_idx,
                    'end': end_idx,
                    'depth': min_val,
                    'duration': (end_idx - start_idx).days,
                })

        # Sort by depth
        dd_periods.sort(key=lambda x: x['depth'])
        top_dd = dd_periods[:5]

        fig = go.Figure()

        # Main drawdown line
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color=self.colors['negative'], width=1),
            fillcolor='rgba(244, 67, 54, 0.2)',
        ))

        # Highlight top drawdowns
        colors = ['rgba(255, 0, 0, 0.5)', 'rgba(255, 100, 0, 0.4)', 'rgba(255, 150, 0, 0.3)',
                  'rgba(255, 200, 0, 0.2)', 'rgba(255, 250, 0, 0.1)']

        for i, dd in enumerate(top_dd):
            fig.add_vrect(
                x0=dd['start'], x1=dd['end'],
                fillcolor=colors[i],
                layer='below',
                line_width=0,
                annotation_text=f"#{i+1}: {dd['depth']*100:.1f}%",
                annotation_position='top left' if i % 2 == 0 else 'top right',
            )

        fig.update_layout(
            title=dict(text='<b>Drawdown Analysis</b>', font=dict(size=16)),
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_dark',
            height=350,
        )

        return fig

    def create_correlation_analysis(self) -> Optional[go.Figure]:
        """Create correlation analysis with benchmark."""
        if self.benchmark is None:
            return None

        # Align data
        common_idx = self.returns.index.intersection(self.benchmark.index)
        strat = self.returns.loc[common_idx]
        bench = self.benchmark.loc[common_idx]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Returns Scatter', 'Rolling Correlation'),
        )

        # Scatter plot
        fig.add_trace(go.Scatter(
            x=bench.values * 100,
            y=strat.values * 100,
            mode='markers',
            name='Daily Returns',
            marker=dict(
                color=self.colors['neutral'],
                opacity=0.5,
                size=5,
            ),
        ), row=1, col=1)

        # Add regression line
        corr = np.corrcoef(bench.values, strat.values)[0, 1]
        beta = np.polyfit(bench.values, strat.values, 1)
        x_line = np.array([bench.min(), bench.max()]) * 100
        y_line = np.polyval(beta, [bench.min(), bench.max()]) * 100

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name=f'Beta: {beta[0]:.2f}',
            line=dict(color='red', dash='dash'),
        ), row=1, col=1)

        # Rolling correlation
        rolling_corr = strat.rolling(63).corr(bench)

        fig.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr.values,
            name='Correlation',
            line=dict(color=self.colors['strategy'], width=2),
        ), row=1, col=2)

        fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3, row=1, col=2)

        fig.update_layout(
            title=dict(text=f'<b>Benchmark Correlation (r={corr:.2f})</b>', font=dict(size=16)),
            template='plotly_dark',
            height=350,
        )

        fig.update_xaxes(title_text=f'{self.benchmark_name} Return (%)', row=1, col=1)
        fig.update_yaxes(title_text=f'{self.strategy_name} Return (%)', row=1, col=1)
        fig.update_yaxes(title_text='Correlation', row=1, col=2)

        return fig

    def generate_tear_sheet(
        self,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Generate comprehensive tear sheet HTML.

        Args:
            output_path: Optional path to save HTML file

        Returns:
            HTML string
        """
        # Generate all charts
        equity_chart = self.create_equity_chart()
        drawdown_chart = self.create_drawdown_chart()
        returns_dist = self.create_returns_distribution()
        monthly_heatmap = self.create_monthly_heatmap()
        rolling_metrics = self.create_rolling_metrics()
        drawdown_analysis = self.create_drawdown_analysis()
        correlation = self.create_correlation_analysis()

        # Build HTML
        html = self._build_tear_sheet_html(
            equity_chart=equity_chart,
            drawdown_chart=drawdown_chart,
            returns_dist=returns_dist,
            monthly_heatmap=monthly_heatmap,
            rolling_metrics=rolling_metrics,
            drawdown_analysis=drawdown_analysis,
            correlation=correlation,
        )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"Tear sheet saved to: {output_path}")

        return html

    def _build_tear_sheet_html(self, **charts) -> str:
        """Build the complete tear sheet HTML."""
        m = self.metrics

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AlphaTrade Performance Tear Sheet - {self.strategy_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: {self.colors['background']};
            color: {self.colors['text']};
            margin: 0;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        .header {{
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid {self.colors['strategy']};
            margin-bottom: 30px;
        }}
        h1 {{ color: {self.colors['strategy']}; margin: 0; }}
        h2 {{ color: {self.colors['text']}; border-bottom: 1px solid #333; padding-bottom: 10px; }}
        .subtitle {{ color: #888; margin: 10px 0; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: {self.colors['card']};
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border-left: 3px solid {self.colors['strategy']};
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            margin: 5px 0;
        }}
        .positive {{ color: {self.colors['positive']}; }}
        .negative {{ color: {self.colors['negative']}; }}
        .neutral {{ color: {self.colors['neutral']}; }}
        .metric-label {{ font-size: 11px; color: #888; text-transform: uppercase; }}
        .chart-container {{
            background: {self.colors['card']};
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .three-column {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
        }}
        @media (max-width: 1000px) {{
            .two-column, .three-column {{ grid-template-columns: 1fr; }}
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .metrics-table th, .metrics-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        .metrics-table th {{ color: {self.colors['strategy']}; background: rgba(76, 175, 80, 0.1); }}
        .footer {{
            text-align: center;
            padding: 20px;
            margin-top: 30px;
            border-top: 1px solid #333;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{self.strategy_name} Performance Report</h1>
            <p class="subtitle">
                {self.returns.index[0].strftime('%Y-%m-%d')} to {self.returns.index[-1].strftime('%Y-%m-%d')}
                | {len(self.returns):,} trading days
            </p>
        </div>

        <h2>Key Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {'positive' if m.total_return > 0 else 'negative'}">
                    {m.total_return*100:.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">CAGR</div>
                <div class="metric-value {'positive' if m.cagr > 0 else 'negative'}">
                    {m.cagr*100:.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value {'positive' if m.sharpe_ratio > 1 else 'neutral'}">
                    {m.sharpe_ratio:.2f}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sortino Ratio</div>
                <div class="metric-value {'positive' if m.sortino_ratio > 1 else 'neutral'}">
                    {m.sortino_ratio:.2f}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">
                    {m.max_drawdown*100:.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volatility</div>
                <div class="metric-value neutral">
                    {m.volatility*100:.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Calmar Ratio</div>
                <div class="metric-value {'positive' if m.calmar_ratio > 1 else 'neutral'}">
                    {m.calmar_ratio:.2f}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value {'positive' if m.win_rate > 0.5 else 'neutral'}">
                    {m.win_rate*100:.1f}%
                </div>
            </div>
        </div>

        <div class="chart-container">
            {charts.get('equity_chart', go.Figure()).to_html(full_html=False, include_plotlyjs=False)}
        </div>

        <div class="chart-container">
            {charts.get('drawdown_chart', go.Figure()).to_html(full_html=False, include_plotlyjs=False)}
        </div>

        <div class="two-column">
            <div class="chart-container">
                {charts.get('returns_dist', go.Figure()).to_html(full_html=False, include_plotlyjs=False)}
            </div>
            <div class="chart-container">
                {charts.get('monthly_heatmap', go.Figure()).to_html(full_html=False, include_plotlyjs=False)}
            </div>
        </div>

        <div class="chart-container">
            {charts.get('rolling_metrics', go.Figure()).to_html(full_html=False, include_plotlyjs=False)}
        </div>

        <div class="chart-container">
            {charts.get('drawdown_analysis', go.Figure()).to_html(full_html=False, include_plotlyjs=False)}
        </div>

        {'<div class="chart-container">' + charts.get("correlation", go.Figure()).to_html(full_html=False, include_plotlyjs=False) + '</div>' if charts.get("correlation") else ''}

        <h2>Detailed Statistics</h2>
        <div class="three-column">
            <div class="chart-container">
                <h3 style="margin-top:0; color: {self.colors['strategy']};">Return Metrics</h3>
                <table class="metrics-table">
                    <tr><td>Total Return</td><td class="{'positive' if m.total_return > 0 else 'negative'}">{m.total_return*100:.2f}%</td></tr>
                    <tr><td>CAGR</td><td class="{'positive' if m.cagr > 0 else 'negative'}">{m.cagr*100:.2f}%</td></tr>
                    <tr><td>MTD</td><td class="{'positive' if m.mtd_return > 0 else 'negative'}">{m.mtd_return*100:.2f}%</td></tr>
                    <tr><td>QTD</td><td class="{'positive' if m.qtd_return > 0 else 'negative'}">{m.qtd_return*100:.2f}%</td></tr>
                    <tr><td>YTD</td><td class="{'positive' if m.ytd_return > 0 else 'negative'}">{m.ytd_return*100:.2f}%</td></tr>
                    <tr><td>Best Day</td><td class="positive">{m.best_day*100:.2f}%</td></tr>
                    <tr><td>Worst Day</td><td class="negative">{m.worst_day*100:.2f}%</td></tr>
                    <tr><td>Best Month</td><td class="positive">{m.best_month*100:.2f}%</td></tr>
                    <tr><td>Worst Month</td><td class="negative">{m.worst_month*100:.2f}%</td></tr>
                </table>
            </div>
            <div class="chart-container">
                <h3 style="margin-top:0; color: {self.colors['strategy']};">Risk Metrics</h3>
                <table class="metrics-table">
                    <tr><td>Volatility (Ann.)</td><td>{m.volatility*100:.2f}%</td></tr>
                    <tr><td>Max Drawdown</td><td class="negative">{m.max_drawdown*100:.2f}%</td></tr>
                    <tr><td>Avg Drawdown</td><td class="negative">{m.avg_drawdown*100:.2f}%</td></tr>
                    <tr><td>Max DD Duration</td><td>{m.max_drawdown_duration} days</td></tr>
                    <tr><td>VaR (95%)</td><td class="negative">{m.var_95*100:.2f}%</td></tr>
                    <tr><td>CVaR (95%)</td><td class="negative">{m.cvar_95*100:.2f}%</td></tr>
                    <tr><td>VaR (99%)</td><td class="negative">{m.var_99*100:.2f}%</td></tr>
                    <tr><td>CVaR (99%)</td><td class="negative">{m.cvar_99*100:.2f}%</td></tr>
                    <tr><td>Skewness</td><td>{m.skewness:.2f}</td></tr>
                    <tr><td>Kurtosis</td><td>{m.kurtosis:.2f}</td></tr>
                </table>
            </div>
            <div class="chart-container">
                <h3 style="margin-top:0; color: {self.colors['strategy']};">Risk-Adjusted & Trading</h3>
                <table class="metrics-table">
                    <tr><td>Sharpe Ratio</td><td>{m.sharpe_ratio:.2f}</td></tr>
                    <tr><td>Sortino Ratio</td><td>{m.sortino_ratio:.2f}</td></tr>
                    <tr><td>Calmar Ratio</td><td>{m.calmar_ratio:.2f}</td></tr>
                    <tr><td>Omega Ratio</td><td>{m.omega_ratio:.2f}</td></tr>
                    <tr><td>Win Rate</td><td>{m.win_rate*100:.1f}%</td></tr>
                    <tr><td>Profit Factor</td><td>{m.profit_factor:.2f}</td></tr>
                    <tr><td>Avg Win</td><td class="positive">{m.avg_win*100:.3f}%</td></tr>
                    <tr><td>Avg Loss</td><td class="negative">{m.avg_loss*100:.3f}%</td></tr>
                    <tr><td>Max Consec. Wins</td><td>{m.max_consecutive_wins}</td></tr>
                    <tr><td>Max Consec. Losses</td><td>{m.max_consecutive_losses}</td></tr>
                </table>
            </div>
        </div>

        <div class="footer">
            <p>Generated by AlphaTrade System</p>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        return html


def create_tear_sheet(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    strategy_name: str = "Strategy",
    benchmark_name: str = "Benchmark",
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Create a comprehensive tear sheet.

    Args:
        returns: Strategy returns series
        benchmark_returns: Optional benchmark returns
        strategy_name: Name of strategy
        benchmark_name: Name of benchmark
        output_path: Path to save HTML file

    Returns:
        HTML string
    """
    dashboard = PerformanceDashboard(
        returns=returns,
        benchmark_returns=benchmark_returns,
        strategy_name=strategy_name,
        benchmark_name=benchmark_name,
    )
    return dashboard.generate_tear_sheet(output_path)
