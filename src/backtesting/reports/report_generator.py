"""
Report generation for AlphaTrade system.

This module provides:
- HTML report generation
- Interactive charts with Plotly
- Performance summaries
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.backtesting.engine import BacktestResult
from src.backtesting.metrics import calculate_sharpe_statistics, SharpeStatistics


class ReportGenerator:
    """
    Comprehensive report generator.

    Creates HTML reports with:
    - Performance summary
    - Interactive charts
    - Trade analysis
    - Risk metrics
    - Institutional-grade Sharpe statistics (DSR, PSR, MinTRL)
    """

    def __init__(
        self,
        result: BacktestResult,
        benchmark_result: BacktestResult | None = None,
        output_dir: Path | str | None = None,
        n_trials: int = 1,
    ) -> None:
        """
        Initialize report generator.

        Args:
            result: Backtest result
            benchmark_result: Optional benchmark result
            output_dir: Output directory for reports
            n_trials: Number of strategies tested (for DSR calculation)
        """
        self.result = result
        self.benchmark = benchmark_result
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_trials = n_trials

        # Calculate institutional Sharpe statistics
        self.sharpe_stats = self._calculate_sharpe_statistics()

    def _calculate_sharpe_statistics(self) -> SharpeStatistics | None:
        """Calculate institutional-grade Sharpe statistics."""
        try:
            if self.result.returns is None or len(self.result.returns) < 50:
                return None

            return calculate_sharpe_statistics(
                returns=self.result.returns,
                n_trials=self.n_trials,
                benchmark_sharpe=0.0,
                risk_free_rate=0.05,
                periods_per_year=252 * 26,  # 15-min bars
            )
        except Exception as e:
            logger.warning(f"Could not calculate Sharpe statistics: {e}")
            return None

    def generate_html(self, filename: str | None = None) -> Path:
        """
        Generate HTML report.

        Args:
            filename: Output filename

        Returns:
            Path to generated report
        """
        if not HAS_PLOTLY:
            logger.warning("Plotly not available. Install with: pip install plotly")
            return self._generate_basic_html(filename)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"backtest_report_{timestamp}.html"
        filepath = self.output_dir / filename

        # Generate charts
        equity_chart = self._create_equity_chart()
        drawdown_chart = self._create_drawdown_chart()
        returns_dist = self._create_returns_distribution()
        monthly_heatmap = self._create_monthly_heatmap()
        rolling_sharpe = self._create_rolling_sharpe()

        # Generate HTML
        html = self._build_html(
            equity_chart=equity_chart,
            drawdown_chart=drawdown_chart,
            returns_dist=returns_dist,
            monthly_heatmap=monthly_heatmap,
            rolling_sharpe=rolling_sharpe,
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Report generated: {filepath}")
        return filepath

    def _create_equity_chart(self) -> str:
        """Create equity curve chart."""
        fig = go.Figure()

        # Strategy equity
        fig.add_trace(go.Scatter(
            x=self.result.equity_curve.index,
            y=self.result.equity_curve.values,
            name="Strategy",
            line=dict(color="blue", width=2),
        ))

        # Benchmark if available
        if self.benchmark:
            fig.add_trace(go.Scatter(
                x=self.benchmark.equity_curve.index,
                y=self.benchmark.equity_curve.values,
                name="Benchmark",
                line=dict(color="gray", width=1, dash="dash"),
            ))

        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark",
            height=400,
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _create_drawdown_chart(self) -> str:
        """Create drawdown chart."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.result.drawdown.index,
            y=self.result.drawdown.values * 100,
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="red"),
            fillcolor="rgba(255, 0, 0, 0.3)",
        ))

        fig.update_layout(
            title="Underwater (Drawdown) Chart",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            height=300,
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _create_returns_distribution(self) -> str:
        """Create returns distribution chart."""
        returns = self.result.returns * 100  # Convert to percentage

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name="Returns",
            marker_color="blue",
            opacity=0.7,
        ))

        # Add normal distribution overlay
        mean = returns.mean()
        std = returns.std()
        x_norm = np.linspace(returns.min(), returns.max(), 100)
        y_norm = (
            len(returns)
            * (returns.max() - returns.min())
            / 50
            * (1 / (std * np.sqrt(2 * np.pi)))
            * np.exp(-((x_norm - mean) ** 2) / (2 * std ** 2))
        )

        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            name="Normal",
            line=dict(color="red", dash="dash"),
        ))

        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=300,
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _create_monthly_heatmap(self) -> str:
        """Create monthly returns heatmap."""
        returns = self.result.returns

        # Calculate monthly returns
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100

        # Create pivot table
        monthly_df = monthly.to_frame("return")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month

        pivot = monthly_df.pivot(index="year", columns="month", values="return")

        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=months[:pivot.shape[1]],
            y=pivot.index,
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate="%{text}%",
            textfont={"size": 10},
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        ))

        fig.update_layout(
            title="Monthly Returns (%)",
            template="plotly_dark",
            height=300,
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _create_rolling_sharpe(self) -> str:
        """Create rolling Sharpe ratio chart."""
        returns = self.result.returns
        window = min(252 * 26, len(returns) // 4)  # Rolling window

        if window < 100:
            return "<p>Insufficient data for rolling Sharpe</p>"

        rolling_mean = returns.rolling(window).mean() * 252 * 26
        rolling_std = returns.rolling(window).std() * np.sqrt(252 * 26)
        rolling_sharpe = rolling_mean / rolling_std

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            name="Rolling Sharpe",
            line=dict(color="green", width=2),
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.3)
        fig.add_hline(y=2, line_dash="dot", line_color="green", opacity=0.3)

        fig.update_layout(
            title=f"Rolling Sharpe Ratio ({window} bars)",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            template="plotly_dark",
            height=300,
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _build_html(self, **charts) -> str:
        """Build the complete HTML report."""
        metrics = self.result.metrics

        # Metrics table
        metrics_html = self._create_metrics_table(metrics)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report - {self.result.metadata.get('strategy_name', 'Strategy')}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #4CAF50;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid #4CAF50;
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .metric-label {{
            font-size: 12px;
            color: #aaa;
            margin-top: 5px;
        }}
        .chart-container {{
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 900px) {{
            .two-column {{
                grid-template-columns: 1fr;
            }}
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            background: #16213e;
            color: #4CAF50;
        }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
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
            <h1>Backtest Report</h1>
            <p>Strategy: {self.result.metadata.get('strategy_name', 'Unknown')}</p>
            <p>Period: {self.result.start_date.strftime('%Y-%m-%d')} to {self.result.end_date.strftime('%Y-%m-%d')}</p>
        </div>

        <h2>Performance Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.get('total_return', 0) > 0 else 'negative'}">
                    {metrics.get('total_return', 0)*100:.2f}%
                </div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.get('cagr', 0) > 0 else 'negative'}">
                    {metrics.get('cagr', 0)*100:.2f}%
                </div>
                <div class="metric-label">CAGR</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('sortino_ratio', 0):.2f}</div>
                <div class="metric-label">Sortino Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{metrics.get('max_drawdown', 0)*100:.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('volatility', 0)*100:.2f}%</div>
                <div class="metric-label">Volatility</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('win_rate', 0)*100:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('profit_factor', 0):.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
        </div>

        <!-- Institutional Sharpe Statistics (JPMorgan-level) -->
        <h2>Statistical Significance (Institutional Metrics)</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {'positive' if self.sharpe_stats and self.sharpe_stats.deflated_sharpe > 0 else 'negative'}">
                    {self.sharpe_stats.deflated_sharpe if self.sharpe_stats else 0:.3f}
                </div>
                <div class="metric-label">Deflated Sharpe Ratio (DSR)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if self.sharpe_stats and self.sharpe_stats.probabilistic_sharpe > 0.95 else ''}">
                    {(self.sharpe_stats.probabilistic_sharpe * 100) if self.sharpe_stats else 0:.1f}%
                </div>
                <div class="metric-label">Probabilistic Sharpe (PSR)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">
                    {self.sharpe_stats.min_track_record_months if self.sharpe_stats and self.sharpe_stats.min_track_record_months < 1000 else 'N/A':.1f}
                </div>
                <div class="metric-label">Min Track Record (months)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if self.sharpe_stats and self.sharpe_stats.is_significant else 'negative'}">
                    {'YES' if self.sharpe_stats and self.sharpe_stats.is_significant else 'NO'}
                </div>
                <div class="metric-label">Statistically Significant</div>
            </div>
        </div>
        <p style="color: #888; font-size: 12px; text-align: center;">
            DSR > 0 indicates skill above random chance. PSR > 95% indicates statistical significance.
            {f'Based on {self.n_trials} strategy trials tested.' if self.n_trials > 1 else ''}
        </p>

        <div class="chart-container">
            <h3>Equity Curve</h3>
            {charts.get('equity_chart', '')}
        </div>

        <div class="chart-container">
            <h3>Drawdown</h3>
            {charts.get('drawdown_chart', '')}
        </div>

        <div class="two-column">
            <div class="chart-container">
                <h3>Returns Distribution</h3>
                {charts.get('returns_dist', '')}
            </div>
            <div class="chart-container">
                <h3>Monthly Returns</h3>
                {charts.get('monthly_heatmap', '')}
            </div>
        </div>

        <div class="chart-container">
            <h3>Rolling Sharpe Ratio</h3>
            {charts.get('rolling_sharpe', '')}
        </div>

        <h2>Detailed Metrics</h2>
        {metrics_html}

        <div class="footer">
            <p>Generated by AlphaTrade Backtesting Engine</p>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _create_metrics_table(self, metrics: dict) -> str:
        """Create HTML metrics table."""
        categories = {
            "Return Metrics": [
                ("Total Return", "total_return", "{:.2%}"),
                ("CAGR", "cagr", "{:.2%}"),
                ("Best Day", "best_day", "{:.2%}"),
                ("Worst Day", "worst_day", "{:.2%}"),
            ],
            "Risk-Adjusted Metrics": [
                ("Sharpe Ratio", "sharpe_ratio", "{:.2f}"),
                ("Sortino Ratio", "sortino_ratio", "{:.2f}"),
                ("Calmar Ratio", "calmar_ratio", "{:.2f}"),
            ],
            "Risk Metrics": [
                ("Max Drawdown", "max_drawdown", "{:.2%}"),
                ("Volatility", "volatility", "{:.2%}"),
                ("VaR (95%)", "var_95", "{:.2%}"),
                ("CVaR (95%)", "cvar_95", "{:.2%}"),
            ],
            "Trading Metrics": [
                ("Win Rate", "win_rate", "{:.1%}"),
                ("Profit Factor", "profit_factor", "{:.2f}"),
                ("Avg Win", "avg_win", "{:.4f}"),
                ("Avg Loss", "avg_loss", "{:.4f}"),
            ],
        }

        html = ""
        for category, items in categories.items():
            html += f"<h3>{category}</h3><table><tr><th>Metric</th><th>Value</th></tr>"
            for label, key, fmt in items:
                value = metrics.get(key, 0)
                if value is not None:
                    formatted = fmt.format(value)
                else:
                    formatted = "N/A"
                html += f"<tr><td>{label}</td><td>{formatted}</td></tr>"
            html += "</table>"

        return html

    def _generate_basic_html(self, filename: str | None) -> Path:
        """Generate basic HTML without Plotly."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"backtest_report_{timestamp}.html"
        filepath = self.output_dir / filename

        metrics = self.result.metrics
        metrics_html = self._create_metrics_table(metrics)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{ font-family: sans-serif; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>Backtest Report</h1>
    <p>Strategy: {self.result.metadata.get('strategy_name', 'Unknown')}</p>
    <p>Period: {self.result.start_date} to {self.result.end_date}</p>
    <p>Final Capital: ${self.result.final_capital:,.2f}</p>
    {metrics_html}
</body>
</html>
"""
        with open(filepath, "w") as f:
            f.write(html)

        return filepath


def generate_html_report(
    result: BacktestResult,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Convenience function to generate HTML report.

    Args:
        result: Backtest result
        output_dir: Output directory

    Returns:
        Path to generated report
    """
    generator = ReportGenerator(result, output_dir=output_dir)
    return generator.generate_html()
