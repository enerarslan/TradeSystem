"""
Pre-Trade Compliance Checks for AlphaTrade System.

JPMorgan-level implementation of pre-trade risk checks:
- Order size limits
- Position concentration limits
- Exposure limits
- Liquidity checks
- Restricted securities

All orders must pass compliance checks before submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class ComplianceResult(Enum):
    """Result of a compliance check."""

    PASSED = "passed"
    REJECTED = "rejected"
    WARNING = "warning"
    MODIFIED = "modified"


@dataclass
class ComplianceCheckResult:
    """Result of a single compliance check."""

    check_name: str
    result: ComplianceResult
    message: str
    original_value: float | None = None
    modified_value: float | None = None
    limit: float | None = None


@dataclass
class Order:
    """Order to be checked for compliance."""

    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float | None = None  # None for market orders
    order_type: str = "MARKET"
    time_in_force: str = "DAY"
    account_id: str = "default"
    strategy_id: str | None = None
    metadata: Dict[str, Any] | None = None


@dataclass
class ComplianceReport:
    """Complete compliance report for an order."""

    order: Order
    timestamp: datetime
    overall_result: ComplianceResult
    checks: List[ComplianceCheckResult]
    modified_quantity: float | None = None
    rejection_reasons: List[str] | None = None


class PreTradeComplianceChecker:
    """
    Pre-trade compliance checking system.

    Validates all orders against risk limits before submission.
    """

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        portfolio_value: float = 1_000_000.0,
    ) -> None:
        """
        Initialize compliance checker.

        Args:
            config: Risk limits configuration
            portfolio_value: Current portfolio value
        """
        self.config = config or {}
        self._portfolio_value = portfolio_value

        # Current positions
        self._positions: Dict[str, Dict[str, Any]] = {}

        # Security master data
        self._security_data: Dict[str, Dict[str, Any]] = {}

        # Restricted list
        self._restricted_securities: set[str] = set()

        # Load limits from config
        trade_limits = self.config.get("trade_limits", {})
        exposure_limits = self.config.get("exposure", {})
        liquidity_limits = self.config.get("liquidity", {})

        self._limits = {
            # Trade size limits
            "min_trade_size": trade_limits.get("min_trade_size", 1000),
            "max_trade_size": trade_limits.get("max_trade_size", 100000),
            "max_daily_trades": trade_limits.get("max_daily_trades", 100),
            "max_daily_turnover": trade_limits.get("max_daily_turnover", 0.50),

            # Position limits
            "max_single_position": exposure_limits.get("max_single_position", 0.20),
            "min_single_position": exposure_limits.get("min_single_position", 0.01),
            "max_gross_exposure": exposure_limits.get("max_gross_exposure", 1.5),
            "max_net_exposure": exposure_limits.get("max_net_exposure", 1.0),
            "max_sector_exposure": exposure_limits.get("max_sector_exposure", 0.40),

            # Liquidity limits
            "max_adv_pct": liquidity_limits.get("max_adv_pct", 0.05),
            "min_adv": liquidity_limits.get("min_adv", 100000),
            "max_spread_pct": liquidity_limits.get("max_spread_pct", 0.50),
        }

        # Tracking
        self._daily_trade_count = 0
        self._daily_turnover = 0.0
        self._last_reset_date: datetime | None = None

        logger.info("Pre-trade compliance checker initialized")

    def update_portfolio_value(self, value: float) -> None:
        """Update current portfolio value."""
        self._portfolio_value = value

    def update_position(
        self,
        symbol: str,
        quantity: float,
        market_value: float,
        sector: str | None = None,
    ) -> None:
        """Update position data."""
        self._positions[symbol] = {
            "quantity": quantity,
            "market_value": market_value,
            "sector": sector,
            "weight": market_value / self._portfolio_value if self._portfolio_value > 0 else 0.0,
        }

    def update_security_data(
        self,
        symbol: str,
        adv: float,
        bid_ask_spread: float = 0.01,
        sector: str | None = None,
        is_restricted: bool = False,
    ) -> None:
        """Update security master data."""
        self._security_data[symbol] = {
            "adv": adv,
            "spread": bid_ask_spread,
            "sector": sector,
            "is_restricted": is_restricted,
        }

        if is_restricted:
            self._restricted_securities.add(symbol)

    def add_restricted_security(self, symbol: str) -> None:
        """Add a security to the restricted list."""
        self._restricted_securities.add(symbol)
        logger.info(f"Added {symbol} to restricted list")

    def remove_restricted_security(self, symbol: str) -> None:
        """Remove a security from the restricted list."""
        self._restricted_securities.discard(symbol)
        logger.info(f"Removed {symbol} from restricted list")

    def check_order(self, order: Order) -> ComplianceReport:
        """
        Run all compliance checks on an order.

        Args:
            order: Order to check

        Returns:
            Compliance report with results of all checks
        """
        self._reset_daily_counters_if_needed()

        timestamp = datetime.now()
        checks: List[ComplianceCheckResult] = []
        rejection_reasons: List[str] = []
        modified_quantity = order.quantity

        # 1. Restricted security check
        restricted_check = self._check_restricted_security(order)
        checks.append(restricted_check)
        if restricted_check.result == ComplianceResult.REJECTED:
            rejection_reasons.append(restricted_check.message)

        # 2. Order size limits
        size_check = self._check_order_size(order)
        checks.append(size_check)
        if size_check.result == ComplianceResult.REJECTED:
            rejection_reasons.append(size_check.message)
        elif size_check.result == ComplianceResult.MODIFIED:
            modified_quantity = size_check.modified_value or modified_quantity

        # 3. Position concentration limit
        concentration_check = self._check_position_concentration(order)
        checks.append(concentration_check)
        if concentration_check.result == ComplianceResult.REJECTED:
            rejection_reasons.append(concentration_check.message)
        elif concentration_check.result == ComplianceResult.MODIFIED:
            modified_quantity = min(
                modified_quantity,
                concentration_check.modified_value or modified_quantity
            )

        # 4. Gross exposure limit
        exposure_check = self._check_gross_exposure(order)
        checks.append(exposure_check)
        if exposure_check.result == ComplianceResult.REJECTED:
            rejection_reasons.append(exposure_check.message)

        # 5. Sector exposure limit
        sector_check = self._check_sector_exposure(order)
        checks.append(sector_check)
        if sector_check.result == ComplianceResult.REJECTED:
            rejection_reasons.append(sector_check.message)
        elif sector_check.result == ComplianceResult.MODIFIED:
            modified_quantity = min(
                modified_quantity,
                sector_check.modified_value or modified_quantity
            )

        # 6. Liquidity check (ADV)
        liquidity_check = self._check_liquidity(order)
        checks.append(liquidity_check)
        if liquidity_check.result == ComplianceResult.REJECTED:
            rejection_reasons.append(liquidity_check.message)
        elif liquidity_check.result == ComplianceResult.MODIFIED:
            modified_quantity = min(
                modified_quantity,
                liquidity_check.modified_value or modified_quantity
            )

        # 7. Daily trade count
        trade_count_check = self._check_daily_trade_count()
        checks.append(trade_count_check)
        if trade_count_check.result == ComplianceResult.REJECTED:
            rejection_reasons.append(trade_count_check.message)

        # 8. Daily turnover
        turnover_check = self._check_daily_turnover(order)
        checks.append(turnover_check)
        if turnover_check.result == ComplianceResult.REJECTED:
            rejection_reasons.append(turnover_check.message)

        # 9. Spread check
        spread_check = self._check_spread(order)
        checks.append(spread_check)
        if spread_check.result == ComplianceResult.WARNING:
            logger.warning(spread_check.message)

        # Determine overall result
        if rejection_reasons:
            overall_result = ComplianceResult.REJECTED
        elif modified_quantity != order.quantity:
            overall_result = ComplianceResult.MODIFIED
        elif any(c.result == ComplianceResult.WARNING for c in checks):
            overall_result = ComplianceResult.WARNING
        else:
            overall_result = ComplianceResult.PASSED

        report = ComplianceReport(
            order=order,
            timestamp=timestamp,
            overall_result=overall_result,
            checks=checks,
            modified_quantity=modified_quantity if modified_quantity != order.quantity else None,
            rejection_reasons=rejection_reasons if rejection_reasons else None,
        )

        # Log result
        if overall_result == ComplianceResult.REJECTED:
            logger.warning(
                f"Order REJECTED: {order.symbol} {order.side} {order.quantity} - "
                f"Reasons: {', '.join(rejection_reasons)}"
            )
        elif overall_result == ComplianceResult.MODIFIED:
            logger.info(
                f"Order MODIFIED: {order.symbol} {order.side} "
                f"{order.quantity} -> {modified_quantity}"
            )
        else:
            logger.debug(f"Order PASSED: {order.symbol} {order.side} {order.quantity}")

        return report

    def _reset_daily_counters_if_needed(self) -> None:
        """Reset daily counters if it's a new day."""
        now = datetime.now()
        if self._last_reset_date is None or now.date() > self._last_reset_date.date():
            self._daily_trade_count = 0
            self._daily_turnover = 0.0
            self._last_reset_date = now
            logger.debug("Daily counters reset")

    def _check_restricted_security(self, order: Order) -> ComplianceCheckResult:
        """Check if security is on restricted list."""
        if order.symbol in self._restricted_securities:
            return ComplianceCheckResult(
                check_name="restricted_security",
                result=ComplianceResult.REJECTED,
                message=f"{order.symbol} is on the restricted securities list",
            )

        return ComplianceCheckResult(
            check_name="restricted_security",
            result=ComplianceResult.PASSED,
            message="Security not restricted",
        )

    def _check_order_size(self, order: Order) -> ComplianceCheckResult:
        """Check order size limits."""
        order_value = order.quantity * (order.price or 0)

        # Estimate order value if no price
        if order_value == 0 and order.symbol in self._positions:
            pos = self._positions[order.symbol]
            price_estimate = pos["market_value"] / pos["quantity"] if pos["quantity"] != 0 else 100
            order_value = order.quantity * price_estimate

        if order_value < self._limits["min_trade_size"]:
            return ComplianceCheckResult(
                check_name="order_size_min",
                result=ComplianceResult.REJECTED,
                message=f"Order value ${order_value:,.0f} below minimum ${self._limits['min_trade_size']:,.0f}",
                original_value=order_value,
                limit=self._limits["min_trade_size"],
            )

        if order_value > self._limits["max_trade_size"]:
            # Modify quantity to fit limit
            price = order.price or (order_value / order.quantity if order.quantity > 0 else 100)
            max_quantity = self._limits["max_trade_size"] / price

            return ComplianceCheckResult(
                check_name="order_size_max",
                result=ComplianceResult.MODIFIED,
                message=f"Order value ${order_value:,.0f} exceeds maximum, reducing quantity",
                original_value=order.quantity,
                modified_value=max_quantity,
                limit=self._limits["max_trade_size"],
            )

        return ComplianceCheckResult(
            check_name="order_size",
            result=ComplianceResult.PASSED,
            message="Order size within limits",
            original_value=order_value,
        )

    def _check_position_concentration(self, order: Order) -> ComplianceCheckResult:
        """Check position concentration limit."""
        current_position = self._positions.get(order.symbol, {})
        current_value = current_position.get("market_value", 0)

        # Estimate new position value
        price = order.price or (
            current_value / current_position.get("quantity", 1)
            if current_position.get("quantity", 0) != 0
            else 100
        )

        if order.side == "BUY":
            new_value = current_value + order.quantity * price
        else:
            new_value = current_value - order.quantity * price

        new_weight = abs(new_value) / self._portfolio_value if self._portfolio_value > 0 else 0

        max_weight = self._limits["max_single_position"]

        if new_weight > max_weight:
            # Calculate max allowable quantity
            max_new_value = max_weight * self._portfolio_value
            max_order_value = abs(max_new_value - current_value)
            max_quantity = max_order_value / price if price > 0 else order.quantity

            return ComplianceCheckResult(
                check_name="position_concentration",
                result=ComplianceResult.MODIFIED,
                message=f"Position weight {new_weight:.1%} would exceed {max_weight:.1%} limit",
                original_value=order.quantity,
                modified_value=max_quantity,
                limit=max_weight,
            )

        return ComplianceCheckResult(
            check_name="position_concentration",
            result=ComplianceResult.PASSED,
            message=f"Position weight {new_weight:.1%} within limit",
            original_value=new_weight,
            limit=max_weight,
        )

    def _check_gross_exposure(self, order: Order) -> ComplianceCheckResult:
        """Check gross exposure limit."""
        current_gross = sum(
            abs(p.get("market_value", 0)) for p in self._positions.values()
        )

        # Estimate order value
        price = order.price or 100
        order_value = order.quantity * price

        if order.side == "BUY":
            new_gross = current_gross + order_value
        else:
            # Selling reduces gross exposure
            current_pos_value = abs(self._positions.get(order.symbol, {}).get("market_value", 0))
            new_gross = current_gross - min(order_value, current_pos_value)

        new_gross_ratio = new_gross / self._portfolio_value if self._portfolio_value > 0 else 0

        if new_gross_ratio > self._limits["max_gross_exposure"]:
            return ComplianceCheckResult(
                check_name="gross_exposure",
                result=ComplianceResult.REJECTED,
                message=f"Gross exposure {new_gross_ratio:.1%} would exceed {self._limits['max_gross_exposure']:.1%}",
                original_value=new_gross_ratio,
                limit=self._limits["max_gross_exposure"],
            )

        return ComplianceCheckResult(
            check_name="gross_exposure",
            result=ComplianceResult.PASSED,
            message=f"Gross exposure {new_gross_ratio:.1%} within limit",
            original_value=new_gross_ratio,
            limit=self._limits["max_gross_exposure"],
        )

    def _check_sector_exposure(self, order: Order) -> ComplianceCheckResult:
        """Check sector exposure limit."""
        # Get sector for this security
        security = self._security_data.get(order.symbol, {})
        sector = security.get("sector") or self._positions.get(order.symbol, {}).get("sector")

        if not sector:
            return ComplianceCheckResult(
                check_name="sector_exposure",
                result=ComplianceResult.WARNING,
                message=f"No sector data for {order.symbol}",
            )

        # Calculate current sector exposure
        sector_exposure = 0.0
        for symbol, pos in self._positions.items():
            pos_sector = pos.get("sector") or self._security_data.get(symbol, {}).get("sector")
            if pos_sector == sector:
                sector_exposure += abs(pos.get("market_value", 0))

        # Add order impact
        price = order.price or 100
        order_value = order.quantity * price

        if order.side == "BUY":
            new_sector_exposure = sector_exposure + order_value
        else:
            new_sector_exposure = max(0, sector_exposure - order_value)

        sector_weight = new_sector_exposure / self._portfolio_value if self._portfolio_value > 0 else 0

        max_sector = self._limits["max_sector_exposure"]

        if sector_weight > max_sector:
            # Calculate max order
            max_sector_value = max_sector * self._portfolio_value
            max_order_value = max_sector_value - sector_exposure
            max_quantity = max_order_value / price if price > 0 else 0

            if max_quantity <= 0:
                return ComplianceCheckResult(
                    check_name="sector_exposure",
                    result=ComplianceResult.REJECTED,
                    message=f"Sector {sector} exposure {sector_weight:.1%} would exceed {max_sector:.1%}",
                    original_value=sector_weight,
                    limit=max_sector,
                )

            return ComplianceCheckResult(
                check_name="sector_exposure",
                result=ComplianceResult.MODIFIED,
                message=f"Reducing order to stay within sector limit",
                original_value=order.quantity,
                modified_value=max_quantity,
                limit=max_sector,
            )

        return ComplianceCheckResult(
            check_name="sector_exposure",
            result=ComplianceResult.PASSED,
            message=f"Sector {sector} exposure {sector_weight:.1%} within limit",
            original_value=sector_weight,
            limit=max_sector,
        )

    def _check_liquidity(self, order: Order) -> ComplianceCheckResult:
        """Check order against ADV limits."""
        security = self._security_data.get(order.symbol, {})
        adv = security.get("adv", 0)

        if adv < self._limits["min_adv"]:
            return ComplianceCheckResult(
                check_name="liquidity_adv",
                result=ComplianceResult.REJECTED,
                message=f"ADV ${adv:,.0f} below minimum ${self._limits['min_adv']:,.0f}",
                original_value=adv,
                limit=self._limits["min_adv"],
            )

        # Check order as percentage of ADV
        price = order.price or 100
        order_value = order.quantity * price

        if adv > 0:
            adv_pct = order_value / adv

            if adv_pct > self._limits["max_adv_pct"]:
                max_order_value = self._limits["max_adv_pct"] * adv
                max_quantity = max_order_value / price if price > 0 else order.quantity

                return ComplianceCheckResult(
                    check_name="liquidity_adv_pct",
                    result=ComplianceResult.MODIFIED,
                    message=f"Order {adv_pct:.1%} of ADV exceeds {self._limits['max_adv_pct']:.1%} limit",
                    original_value=order.quantity,
                    modified_value=max_quantity,
                    limit=self._limits["max_adv_pct"],
                )

        return ComplianceCheckResult(
            check_name="liquidity",
            result=ComplianceResult.PASSED,
            message="Liquidity check passed",
        )

    def _check_daily_trade_count(self) -> ComplianceCheckResult:
        """Check daily trade count limit."""
        if self._daily_trade_count >= self._limits["max_daily_trades"]:
            return ComplianceCheckResult(
                check_name="daily_trade_count",
                result=ComplianceResult.REJECTED,
                message=f"Daily trade count {self._daily_trade_count} at limit {self._limits['max_daily_trades']}",
                original_value=float(self._daily_trade_count),
                limit=float(self._limits["max_daily_trades"]),
            )

        return ComplianceCheckResult(
            check_name="daily_trade_count",
            result=ComplianceResult.PASSED,
            message=f"Daily trades: {self._daily_trade_count} / {self._limits['max_daily_trades']}",
        )

    def _check_daily_turnover(self, order: Order) -> ComplianceCheckResult:
        """Check daily turnover limit."""
        price = order.price or 100
        order_value = order.quantity * price

        new_turnover = self._daily_turnover + order_value
        turnover_pct = new_turnover / self._portfolio_value if self._portfolio_value > 0 else 0

        if turnover_pct > self._limits["max_daily_turnover"]:
            return ComplianceCheckResult(
                check_name="daily_turnover",
                result=ComplianceResult.REJECTED,
                message=f"Daily turnover {turnover_pct:.1%} would exceed {self._limits['max_daily_turnover']:.1%}",
                original_value=turnover_pct,
                limit=self._limits["max_daily_turnover"],
            )

        return ComplianceCheckResult(
            check_name="daily_turnover",
            result=ComplianceResult.PASSED,
            message=f"Daily turnover: {turnover_pct:.1%} / {self._limits['max_daily_turnover']:.1%}",
        )

    def _check_spread(self, order: Order) -> ComplianceCheckResult:
        """Check bid-ask spread for market orders."""
        if order.order_type != "MARKET":
            return ComplianceCheckResult(
                check_name="spread",
                result=ComplianceResult.PASSED,
                message="Not a market order",
            )

        security = self._security_data.get(order.symbol, {})
        spread = security.get("spread", 0)

        if spread > self._limits["max_spread_pct"]:
            return ComplianceCheckResult(
                check_name="spread",
                result=ComplianceResult.WARNING,
                message=f"Spread {spread:.2%} exceeds {self._limits['max_spread_pct']:.2%} - consider limit order",
                original_value=spread,
                limit=self._limits["max_spread_pct"],
            )

        return ComplianceCheckResult(
            check_name="spread",
            result=ComplianceResult.PASSED,
            message=f"Spread {spread:.2%} within limit",
        )

    def record_executed_trade(self, order: Order) -> None:
        """Record an executed trade for daily tracking."""
        self._daily_trade_count += 1
        price = order.price or 100
        self._daily_turnover += order.quantity * price

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get summary of compliance state."""
        return {
            "portfolio_value": self._portfolio_value,
            "position_count": len(self._positions),
            "daily_trade_count": self._daily_trade_count,
            "daily_trade_limit": self._limits["max_daily_trades"],
            "daily_turnover": self._daily_turnover,
            "daily_turnover_limit_pct": self._limits["max_daily_turnover"],
            "restricted_securities": len(self._restricted_securities),
            "limits": self._limits,
        }
