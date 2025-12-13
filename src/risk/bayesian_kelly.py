"""
Bayesian Kelly Position Sizer
=============================
JPMorgan-Level Uncertainty-Aware Position Sizing

The standard Kelly criterion assumes known win rate and payoff ratio.
In practice, these are ESTIMATED with uncertainty. This module:

1. Models parameter uncertainty with Bayesian posteriors
2. Computes Kelly fraction accounting for estimation error
3. Reduces position size when uncertain
4. Updates beliefs as new data arrives

Key Insight: When uncertain about edge, bet smaller.
A trader with 55% estimated win rate but wide confidence interval
should bet less than one with 55% and narrow confidence interval.

Mathematical Foundation:
- Prior: Beta(α, β) for win rate
- Likelihood: Binomial(n, k) for wins/losses
- Posterior: Beta(α + k, β + n - k)
- Kelly under uncertainty: E[f*] with variance penalty

Author: AlphaTrade Institutional System
Based on: ARCHITECTURAL_REVIEW_REPORT.md - Phase 3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize_scalar

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeOutcome:
    """Record of a trade outcome for belief updating"""
    symbol: str
    strategy: str
    win: bool
    profit_pct: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BayesianEstimate:
    """Bayesian estimate with uncertainty"""
    mean: float
    std: float
    lower_95: float
    upper_95: float
    n_observations: int

    def confidence_width(self) -> float:
        """Width of 95% CI as fraction of mean"""
        if self.mean == 0:
            return float('inf')
        return (self.upper_95 - self.lower_95) / abs(self.mean)


@dataclass
class KellyResult:
    """Result of Kelly calculation"""
    kelly_fraction: float  # Raw Kelly
    bayesian_fraction: float  # Adjusted for uncertainty
    fractional_kelly: float  # Final recommended (with safety factor)
    win_rate_estimate: BayesianEstimate
    payoff_ratio_estimate: BayesianEstimate
    edge_estimate: BayesianEstimate
    uncertainty_penalty: float
    max_position_pct: float  # Capped position

    def to_dict(self) -> Dict[str, Any]:
        return {
            'kelly_fraction': self.kelly_fraction,
            'bayesian_fraction': self.bayesian_fraction,
            'fractional_kelly': self.fractional_kelly,
            'win_rate': self.win_rate_estimate.mean,
            'win_rate_std': self.win_rate_estimate.std,
            'payoff_ratio': self.payoff_ratio_estimate.mean,
            'edge': self.edge_estimate.mean,
            'uncertainty_penalty': self.uncertainty_penalty,
            'max_position_pct': self.max_position_pct
        }


class BayesianKellySizer:
    """
    Position sizing with Bayesian uncertainty estimation.

    Standard Kelly: f* = (bp - q) / b
    where:
        p = win probability
        q = lose probability (1-p)
        b = payoff ratio (avg_win / avg_loss)

    Bayesian Kelly accounts for:
    1. Uncertainty in p (win rate)
    2. Uncertainty in b (payoff ratio)
    3. Estimation error penalty

    When uncertain, bet less. As evidence accumulates, bet more.
    """

    def __init__(
        self,
        # Prior parameters (uninformative by default)
        prior_wins: float = 2.0,  # Beta prior alpha
        prior_losses: float = 2.0,  # Beta prior beta
        prior_avg_win: float = 0.02,  # Prior avg win %
        prior_avg_loss: float = 0.02,  # Prior avg loss %
        prior_variance: float = 0.01,  # Prior variance for payoffs

        # Kelly adjustments
        kelly_fraction: float = 0.25,  # Fractional Kelly (25% is common)
        max_position_pct: float = 0.20,  # Maximum 20% of portfolio
        min_observations: int = 20,  # Minimum trades before full Kelly

        # Uncertainty handling
        uncertainty_penalty_weight: float = 1.0,  # How much to penalize uncertainty
        min_edge_threshold: float = 0.01,  # Minimum edge to trade
    ):
        # Prior parameters
        self.prior_wins = prior_wins
        self.prior_losses = prior_losses
        self.prior_avg_win = prior_avg_win
        self.prior_avg_loss = prior_avg_loss
        self.prior_variance = prior_variance

        # Kelly parameters
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_observations = min_observations

        # Uncertainty
        self.uncertainty_penalty_weight = uncertainty_penalty_weight
        self.min_edge_threshold = min_edge_threshold

        # Track outcomes by strategy
        self._outcomes: Dict[str, List[TradeOutcome]] = {}

        # Cached posteriors
        self._posteriors: Dict[str, Dict] = {}

    def record_outcome(self, outcome: TradeOutcome) -> None:
        """
        Record a trade outcome to update beliefs.

        As we observe more trades, our uncertainty decreases
        and position sizes can increase.
        """
        key = f"{outcome.symbol}_{outcome.strategy}"

        if key not in self._outcomes:
            self._outcomes[key] = []

        self._outcomes[key].append(outcome)

        # Invalidate cached posterior
        if key in self._posteriors:
            del self._posteriors[key]

    def calculate_kelly(
        self,
        symbol: str,
        strategy: str,
        signal_strength: float = 1.0
    ) -> KellyResult:
        """
        Calculate Bayesian Kelly fraction.

        Args:
            symbol: Trading symbol
            strategy: Strategy name
            signal_strength: Signal strength (0-1), scales position

        Returns:
            KellyResult with position sizing recommendations
        """
        key = f"{symbol}_{strategy}"

        # Get or compute posterior
        if key not in self._posteriors:
            self._posteriors[key] = self._compute_posterior(key)

        posterior = self._posteriors[key]

        # Extract estimates
        win_rate = posterior['win_rate']
        payoff_ratio = posterior['payoff_ratio']

        # Compute edge estimate
        edge_mean = win_rate.mean * payoff_ratio.mean - (1 - win_rate.mean)
        edge_std = self._compute_edge_std(win_rate, payoff_ratio)

        edge = BayesianEstimate(
            mean=edge_mean,
            std=edge_std,
            lower_95=edge_mean - 1.96 * edge_std,
            upper_95=edge_mean + 1.96 * edge_std,
            n_observations=win_rate.n_observations
        )

        # Check minimum edge
        if edge.mean < self.min_edge_threshold:
            return KellyResult(
                kelly_fraction=0.0,
                bayesian_fraction=0.0,
                fractional_kelly=0.0,
                win_rate_estimate=win_rate,
                payoff_ratio_estimate=payoff_ratio,
                edge_estimate=edge,
                uncertainty_penalty=1.0,
                max_position_pct=0.0
            )

        # Raw Kelly
        if payoff_ratio.mean > 0:
            raw_kelly = (win_rate.mean * payoff_ratio.mean - (1 - win_rate.mean)) / payoff_ratio.mean
        else:
            raw_kelly = 0.0

        # Bayesian Kelly with uncertainty penalty
        # Key insight: Wide confidence intervals → reduce bet
        uncertainty_penalty = self._compute_uncertainty_penalty(win_rate, payoff_ratio, edge)

        bayesian_kelly = raw_kelly * (1 - uncertainty_penalty)

        # Apply fractional Kelly
        fractional = bayesian_kelly * self.kelly_fraction * signal_strength

        # Cap at maximum
        capped = min(max(fractional, 0), self.max_position_pct)

        # Scale by observations (ramp up as we get data)
        n_obs = win_rate.n_observations
        if n_obs < self.min_observations:
            # Linear ramp from 0 to full at min_observations
            ramp_factor = n_obs / self.min_observations
            capped = capped * ramp_factor

        return KellyResult(
            kelly_fraction=raw_kelly,
            bayesian_fraction=bayesian_kelly,
            fractional_kelly=capped,
            win_rate_estimate=win_rate,
            payoff_ratio_estimate=payoff_ratio,
            edge_estimate=edge,
            uncertainty_penalty=uncertainty_penalty,
            max_position_pct=capped
        )

    def _compute_posterior(self, key: str) -> Dict[str, BayesianEstimate]:
        """Compute posterior distributions from observed outcomes"""
        outcomes = self._outcomes.get(key, [])

        # Win rate posterior (Beta distribution)
        wins = sum(1 for o in outcomes if o.win)
        losses = len(outcomes) - wins

        # Posterior parameters
        alpha = self.prior_wins + wins
        beta = self.prior_losses + losses

        # Beta distribution statistics
        win_rate_mean = alpha / (alpha + beta)
        win_rate_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        win_rate_std = np.sqrt(win_rate_var)

        # 95% credible interval
        win_rate_lower = stats.beta.ppf(0.025, alpha, beta)
        win_rate_upper = stats.beta.ppf(0.975, alpha, beta)

        win_rate_estimate = BayesianEstimate(
            mean=win_rate_mean,
            std=win_rate_std,
            lower_95=win_rate_lower,
            upper_95=win_rate_upper,
            n_observations=len(outcomes)
        )

        # Payoff ratio posterior (Normal-Inverse-Gamma for conjugate)
        # Simplified: use sample mean/std with prior weighting
        if outcomes:
            win_profits = [o.profit_pct for o in outcomes if o.win and o.profit_pct > 0]
            loss_profits = [abs(o.profit_pct) for o in outcomes if not o.win and o.profit_pct < 0]

            # Average win
            if win_profits:
                avg_win = np.mean(win_profits)
                win_std = np.std(win_profits) / np.sqrt(len(win_profits)) if len(win_profits) > 1 else self.prior_variance
            else:
                avg_win = self.prior_avg_win
                win_std = np.sqrt(self.prior_variance)

            # Average loss
            if loss_profits:
                avg_loss = np.mean(loss_profits)
                loss_std = np.std(loss_profits) / np.sqrt(len(loss_profits)) if len(loss_profits) > 1 else self.prior_variance
            else:
                avg_loss = self.prior_avg_loss
                loss_std = np.sqrt(self.prior_variance)

            # Payoff ratio
            if avg_loss > 0:
                payoff_mean = avg_win / avg_loss
                # Delta method for ratio variance
                payoff_var = (win_std / avg_loss) ** 2 + (avg_win * loss_std / avg_loss ** 2) ** 2
                payoff_std = np.sqrt(payoff_var)
            else:
                payoff_mean = self.prior_avg_win / self.prior_avg_loss
                payoff_std = np.sqrt(self.prior_variance)

        else:
            payoff_mean = self.prior_avg_win / self.prior_avg_loss
            payoff_std = np.sqrt(self.prior_variance)

        payoff_estimate = BayesianEstimate(
            mean=payoff_mean,
            std=payoff_std,
            lower_95=payoff_mean - 1.96 * payoff_std,
            upper_95=payoff_mean + 1.96 * payoff_std,
            n_observations=len(outcomes)
        )

        return {
            'win_rate': win_rate_estimate,
            'payoff_ratio': payoff_estimate
        }

    def _compute_edge_std(
        self,
        win_rate: BayesianEstimate,
        payoff_ratio: BayesianEstimate
    ) -> float:
        """
        Compute standard deviation of edge estimate.

        Edge = p * b - (1 - p) = p * (b + 1) - 1

        Var(Edge) ≈ (b+1)² * Var(p) + p² * Var(b) + 2*p*(b+1)*Cov(p,b)
        Assuming independence: Cov(p,b) = 0
        """
        p = win_rate.mean
        b = payoff_ratio.mean
        var_p = win_rate.std ** 2
        var_b = payoff_ratio.std ** 2

        var_edge = (b + 1) ** 2 * var_p + p ** 2 * var_b

        return np.sqrt(var_edge)

    def _compute_uncertainty_penalty(
        self,
        win_rate: BayesianEstimate,
        payoff_ratio: BayesianEstimate,
        edge: BayesianEstimate
    ) -> float:
        """
        Compute penalty for parameter uncertainty.

        Higher uncertainty → higher penalty → smaller bet

        Uses coefficient of variation (CV) of edge estimate.
        """
        if edge.mean <= 0:
            return 1.0  # Full penalty (no betting)

        # Coefficient of variation
        cv = edge.std / edge.mean

        # Also consider if lower CI bound is negative
        if edge.lower_95 < 0:
            # If 95% CI includes 0 or negative, heavy penalty
            prob_negative = stats.norm.cdf(0, edge.mean, edge.std)
            cv += prob_negative

        # Penalty formula: 1 - exp(-weight * cv)
        # CV = 0 → penalty = 0
        # CV = ∞ → penalty = 1
        penalty = 1 - np.exp(-self.uncertainty_penalty_weight * cv)

        return min(max(penalty, 0), 0.9)  # Cap at 90% penalty

    def get_position_size(
        self,
        symbol: str,
        strategy: str,
        portfolio_value: float,
        current_price: float,
        signal_strength: float = 1.0
    ) -> Tuple[int, KellyResult]:
        """
        Get recommended position size in shares.

        Args:
            symbol: Trading symbol
            strategy: Strategy name
            portfolio_value: Total portfolio value
            current_price: Current price per share
            signal_strength: Signal strength (0-1)

        Returns:
            Tuple of (shares, KellyResult)
        """
        result = self.calculate_kelly(symbol, strategy, signal_strength)

        # Calculate dollar amount
        dollar_amount = portfolio_value * result.fractional_kelly

        # Convert to shares
        if current_price > 0:
            shares = int(dollar_amount / current_price)
        else:
            shares = 0

        return shares, result

    def get_statistics(self, symbol: str = None, strategy: str = None) -> Dict[str, Any]:
        """Get statistics for symbol/strategy combination"""
        if symbol and strategy:
            key = f"{symbol}_{strategy}"
            outcomes = self._outcomes.get(key, [])

            if not outcomes:
                return {'n_trades': 0}

            wins = [o for o in outcomes if o.win]
            losses = [o for o in outcomes if not o.win]

            return {
                'n_trades': len(outcomes),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(outcomes),
                'avg_win_pct': np.mean([o.profit_pct for o in wins]) if wins else 0,
                'avg_loss_pct': np.mean([o.profit_pct for o in losses]) if losses else 0,
                'profit_factor': (
                    abs(sum(o.profit_pct for o in wins)) /
                    abs(sum(o.profit_pct for o in losses))
                    if losses and sum(o.profit_pct for o in losses) != 0 else float('inf')
                )
            }
        else:
            # Overall statistics
            all_outcomes = []
            for outcomes in self._outcomes.values():
                all_outcomes.extend(outcomes)

            if not all_outcomes:
                return {'n_trades': 0}

            wins = [o for o in all_outcomes if o.win]
            return {
                'n_trades': len(all_outcomes),
                'wins': len(wins),
                'win_rate': len(wins) / len(all_outcomes),
                'strategies_tracked': len(self._outcomes)
            }

    def reset_beliefs(self, symbol: str = None, strategy: str = None) -> None:
        """Reset beliefs (start fresh)"""
        if symbol and strategy:
            key = f"{symbol}_{strategy}"
            self._outcomes.pop(key, None)
            self._posteriors.pop(key, None)
        else:
            self._outcomes.clear()
            self._posteriors.clear()


# =============================================================================
# INTEGRATION WITH EXISTING POSITION SIZER
# =============================================================================

class BayesianKellyPositionSizer:
    """
    Integration class matching existing position sizer interface.

    Wraps BayesianKellySizer to match the interface expected by
    the trading system.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.20,
        target_volatility: float = 0.15
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.target_volatility = target_volatility

        self._kelly = BayesianKellySizer(
            kelly_fraction=kelly_fraction,
            max_position_pct=max_position_pct
        )

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        signal_strength: float,
        volatility: float = None,
        strategy_name: str = "default"
    ):
        """
        Calculate position size.

        Returns object with:
        - shares: Number of shares
        - dollars: Dollar amount
        - pct_portfolio: Percentage of portfolio
        """
        shares, kelly_result = self._kelly.get_position_size(
            symbol=symbol,
            strategy=strategy_name,
            portfolio_value=portfolio_value,
            current_price=current_price,
            signal_strength=abs(signal_strength)
        )

        # Adjust for volatility if provided
        if volatility and volatility > 0:
            vol_adjustment = self.target_volatility / volatility
            shares = int(shares * min(vol_adjustment, 2.0))

        dollars = shares * current_price
        pct_portfolio = dollars / portfolio_value if portfolio_value > 0 else 0

        # Return named tuple-like object
        class PositionSize:
            pass

        result = PositionSize()
        result.shares = shares
        result.dollars = dollars
        result.pct_portfolio = pct_portfolio
        result.kelly_result = kelly_result

        return result

    def record_trade(
        self,
        symbol: str,
        strategy: str,
        entry_price: float,
        exit_price: float,
        side: str  # 'long' or 'short'
    ) -> None:
        """Record a completed trade for learning"""
        if side == 'long':
            profit_pct = (exit_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - exit_price) / entry_price

        outcome = TradeOutcome(
            symbol=symbol,
            strategy=strategy,
            win=profit_pct > 0,
            profit_pct=profit_pct
        )

        self._kelly.record_outcome(outcome)
