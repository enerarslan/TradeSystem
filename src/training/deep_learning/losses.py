"""
Custom financial loss functions for deep learning.

This module provides differentiable loss functions for optimizing
financial objectives like Sharpe ratio, Sortino ratio, and drawdown.

These losses can be used directly with PyTorch models to train
for risk-adjusted returns rather than just prediction accuracy.

Reference:
    "Deep Learning for Portfolio Optimization" - Zhang et al. (2020)
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:

    class SharpeLoss(nn.Module):
        """
        Negative Sharpe ratio as a loss function.

        Maximizes risk-adjusted returns by minimizing negative Sharpe.

        Loss = -mean(strategy_returns) / (std(strategy_returns) + eps)

        Args:
            eps: Small constant to prevent division by zero
            annualization_factor: Factor to annualize Sharpe (sqrt of periods per year)
        """

        def __init__(
            self,
            eps: float = 1e-8,
            annualization_factor: float = 1.0,
        ):
            super().__init__()
            self.eps = eps
            self.annualization_factor = annualization_factor

        def forward(
            self,
            predictions: torch.Tensor,
            returns: torch.Tensor,
        ) -> torch.Tensor:
            """
            Calculate negative Sharpe ratio.

            Args:
                predictions: Model predictions/signals (batch_size, 1) or (batch_size,)
                returns: Asset returns for the same period (batch_size, 1) or (batch_size,)

            Returns:
                Negative Sharpe ratio (scalar)
            """
            # Ensure same shape
            predictions = predictions.view(-1)
            returns = returns.view(-1)

            # Normalize predictions to positions [-1, 1]
            positions = torch.tanh(predictions)

            # Strategy returns = position * asset returns
            strategy_returns = positions * returns

            # Calculate Sharpe
            mean_return = strategy_returns.mean()
            std_return = strategy_returns.std() + self.eps

            sharpe = mean_return / std_return * self.annualization_factor

            # Return negative (we minimize loss, want to maximize Sharpe)
            return -sharpe


    class SortinoLoss(nn.Module):
        """
        Negative Sortino ratio as a loss function.

        Similar to Sharpe but only penalizes downside volatility.

        Loss = -mean(strategy_returns) / (downside_std + eps)
        """

        def __init__(
            self,
            target_return: float = 0.0,
            eps: float = 1e-8,
            annualization_factor: float = 1.0,
        ):
            super().__init__()
            self.target_return = target_return
            self.eps = eps
            self.annualization_factor = annualization_factor

        def forward(
            self,
            predictions: torch.Tensor,
            returns: torch.Tensor,
        ) -> torch.Tensor:
            """Calculate negative Sortino ratio."""
            predictions = predictions.view(-1)
            returns = returns.view(-1)

            positions = torch.tanh(predictions)
            strategy_returns = positions * returns

            mean_return = strategy_returns.mean()

            # Downside returns (below target)
            downside = torch.clamp(self.target_return - strategy_returns, min=0)
            downside_std = torch.sqrt((downside ** 2).mean()) + self.eps

            sortino = mean_return / downside_std * self.annualization_factor

            return -sortino


    class MaxDrawdownLoss(nn.Module):
        """
        Maximum drawdown penalty loss.

        Penalizes strategies with large drawdowns.

        Loss = max_drawdown * penalty_weight
        """

        def __init__(
            self,
            penalty_weight: float = 1.0,
            eps: float = 1e-8,
        ):
            super().__init__()
            self.penalty_weight = penalty_weight
            self.eps = eps

        def forward(
            self,
            predictions: torch.Tensor,
            returns: torch.Tensor,
        ) -> torch.Tensor:
            """Calculate max drawdown penalty."""
            predictions = predictions.view(-1)
            returns = returns.view(-1)

            positions = torch.tanh(predictions)
            strategy_returns = positions * returns

            # Cumulative returns (cumulative sum for log-like returns)
            cumulative = torch.cumsum(strategy_returns, dim=0)

            # Running maximum
            running_max = torch.cummax(cumulative, dim=0)[0]

            # Drawdown
            drawdown = running_max - cumulative

            # Max drawdown
            max_dd = drawdown.max()

            return max_dd * self.penalty_weight


    class CalmarLoss(nn.Module):
        """
        Negative Calmar ratio as loss.

        Calmar = Annualized Return / Max Drawdown
        """

        def __init__(
            self,
            annualization_factor: float = 252.0,
            eps: float = 1e-8,
        ):
            super().__init__()
            self.annualization_factor = annualization_factor
            self.eps = eps

        def forward(
            self,
            predictions: torch.Tensor,
            returns: torch.Tensor,
        ) -> torch.Tensor:
            """Calculate negative Calmar ratio."""
            predictions = predictions.view(-1)
            returns = returns.view(-1)

            positions = torch.tanh(predictions)
            strategy_returns = positions * returns

            # Annualized return
            mean_return = strategy_returns.mean() * self.annualization_factor

            # Max drawdown
            cumulative = torch.cumsum(strategy_returns, dim=0)
            running_max = torch.cummax(cumulative, dim=0)[0]
            drawdown = running_max - cumulative
            max_dd = drawdown.max() + self.eps

            calmar = mean_return / max_dd

            return -calmar


    class CombinedFinancialLoss(nn.Module):
        """
        Combined loss function with multiple financial objectives.

        Weighted combination of:
        - Sharpe ratio (risk-adjusted return)
        - Sortino ratio (downside risk)
        - Max drawdown penalty
        - Optional MSE/BCE for prediction accuracy

        Example:
            loss_fn = CombinedFinancialLoss(
                sharpe_weight=0.4,
                sortino_weight=0.3,
                drawdown_weight=0.2,
                accuracy_weight=0.1
            )
        """

        def __init__(
            self,
            sharpe_weight: float = 0.5,
            sortino_weight: float = 0.3,
            drawdown_weight: float = 0.2,
            accuracy_weight: float = 0.0,
            annualization_factor: float = 1.0,
            eps: float = 1e-8,
        ):
            super().__init__()

            self.sharpe_weight = sharpe_weight
            self.sortino_weight = sortino_weight
            self.drawdown_weight = drawdown_weight
            self.accuracy_weight = accuracy_weight

            self.sharpe_loss = SharpeLoss(eps, annualization_factor)
            self.sortino_loss = SortinoLoss(0.0, eps, annualization_factor)
            self.drawdown_loss = MaxDrawdownLoss(1.0, eps)

        def forward(
            self,
            predictions: torch.Tensor,
            returns: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Calculate combined loss.

            Args:
                predictions: Model predictions
                returns: Asset returns
                targets: Optional ground truth targets for accuracy loss

            Returns:
                Combined loss value
            """
            total_loss = torch.tensor(0.0, device=predictions.device)

            if self.sharpe_weight > 0:
                total_loss += self.sharpe_weight * self.sharpe_loss(predictions, returns)

            if self.sortino_weight > 0:
                total_loss += self.sortino_weight * self.sortino_loss(predictions, returns)

            if self.drawdown_weight > 0:
                total_loss += self.drawdown_weight * self.drawdown_loss(predictions, returns)

            if self.accuracy_weight > 0 and targets is not None:
                accuracy_loss = F.mse_loss(predictions.view(-1), targets.view(-1))
                total_loss += self.accuracy_weight * accuracy_loss

            return total_loss


    class DirectionalAccuracyLoss(nn.Module):
        """
        Loss function that penalizes incorrect direction predictions.

        Useful when the sign of the prediction (buy/sell) matters
        more than the magnitude.
        """

        def __init__(self, weight: float = 1.0):
            super().__init__()
            self.weight = weight

        def forward(
            self,
            predictions: torch.Tensor,
            returns: torch.Tensor,
        ) -> torch.Tensor:
            """Calculate directional accuracy loss."""
            predictions = predictions.view(-1)
            returns = returns.view(-1)

            # Signs match when product is positive
            correct_direction = (predictions * returns) > 0

            # Loss is proportion of incorrect predictions
            accuracy = correct_direction.float().mean()

            return (1 - accuracy) * self.weight


    class ProfitFactorLoss(nn.Module):
        """
        Negative profit factor as loss.

        Profit Factor = Gross Profits / Gross Losses
        """

        def __init__(self, eps: float = 1e-8):
            super().__init__()
            self.eps = eps

        def forward(
            self,
            predictions: torch.Tensor,
            returns: torch.Tensor,
        ) -> torch.Tensor:
            """Calculate negative profit factor."""
            predictions = predictions.view(-1)
            returns = returns.view(-1)

            positions = torch.tanh(predictions)
            strategy_returns = positions * returns

            gross_profits = torch.clamp(strategy_returns, min=0).sum()
            gross_losses = torch.abs(torch.clamp(strategy_returns, max=0).sum())

            profit_factor = gross_profits / (gross_losses + self.eps)

            return -profit_factor

else:
    # Dummy classes when PyTorch not available
    class SharpeLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for SharpeLoss")

    class SortinoLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for SortinoLoss")

    class MaxDrawdownLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for MaxDrawdownLoss")

    class CombinedFinancialLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for CombinedFinancialLoss")
