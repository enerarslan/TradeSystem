"""
Institutional Risk Management System
JPMorgan-Level Pre-Trade and Real-Time Risk Controls

Features:
- Pre-trade risk validation
- Real-time risk monitoring
- VaR and CVaR calculations
- Position limits and concentration checks
- Circuit breakers
- Correlation risk monitoring
- Stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import threading
from scipy.stats import norm, t as student_t
from scipy.optimize import minimize

from ..utils.logger import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    BREACH = "breach"


class RiskType(Enum):
    """Types of risk"""
    MARKET = "market"
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"
    CORRELATION = "correlation"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    LEVERAGE = "leverage"
    COUNTERPARTY = "counterparty"


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    # Position limits
    max_position_pct: float = 0.10  # Max 10% per position
    max_sector_pct: float = 0.30  # Max 30% per sector
    max_single_name_notional: float = 500000  # Max $500K per name
    max_total_notional: float = 5000000  # Max $5M total

    # Portfolio limits
    max_leverage: float = 1.0  # No leverage
    max_gross_exposure: float = 1.5  # 150% gross
    max_net_exposure: float = 1.0  # 100% net
    max_beta_exposure: float = 1.2  # Max portfolio beta

    # Risk limits
    max_var_pct: float = 0.02  # 2% daily VaR limit
    max_cvar_pct: float = 0.03  # 3% daily CVaR limit
    max_drawdown: float = 0.15  # 15% max drawdown
    max_daily_loss: float = 0.05  # 5% daily loss limit
    max_weekly_loss: float = 0.10  # 10% weekly loss limit

    # Volatility limits
    target_volatility: float = 0.15  # 15% annual vol target
    max_volatility: float = 0.25  # 25% max volatility
    vol_scaling_threshold: float = 0.20  # Scale at 20%

    # Correlation limits
    max_correlated_exposure: float = 0.40  # 40% in correlated assets
    correlation_threshold: float = 0.70  # Considered correlated above 70%

    # Trading limits
    max_daily_trades: int = 100
    max_order_size_pct: float = 0.05  # 5% of ADV
    max_slippage_bps: float = 20  # 20 bps slippage limit

    # Circuit breakers
    circuit_breaker_loss: float = 0.03  # 3% intraday loss
    circuit_breaker_duration: int = 30  # Minutes to pause


@dataclass
class RiskMetrics:
    """Real-time risk metrics"""
    timestamp: datetime
    portfolio_value: float
    cash: float

    # Exposure metrics
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    leverage: float

    # Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    portfolio_volatility: float
    portfolio_beta: float

    # Drawdown metrics
    current_drawdown: float
    max_drawdown: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float

    # Concentration metrics
    largest_position_pct: float
    largest_sector_pct: float
    herfindahl_index: float
    correlation_risk: float

    # Trading metrics
    daily_trades: int
    daily_turnover: float
    avg_slippage_bps: float

    # Risk level
    overall_risk_level: RiskLevel = RiskLevel.LOW
    risk_alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'gross_exposure': self.gross_exposure,
            'net_exposure': self.net_exposure,
            'leverage': self.leverage,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'portfolio_volatility': self.portfolio_volatility,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'daily_pnl': self.daily_pnl,
            'overall_risk_level': self.overall_risk_level.value,
            'risk_alerts': self.risk_alerts
        }


@dataclass
class PreTradeRiskCheck:
    """Pre-trade risk check result"""
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    notional: float

    # Check results
    passed: bool
    risk_score: float  # 0-100
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Adjusted values (if order was modified)
    adjusted_quantity: Optional[int] = None
    adjustment_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'notional': self.notional,
            'passed': self.passed,
            'risk_score': self.risk_score,
            'checks_passed': self.checks_passed,
            'checks_failed': self.checks_failed,
            'warnings': self.warnings,
            'adjusted_quantity': self.adjusted_quantity
        }


class VaRCalculator:
    """
    Value at Risk calculator using multiple methods.

    Methods:
    - Historical simulation
    - Parametric (variance-covariance)
    - Monte Carlo simulation
    - Cornish-Fisher expansion
    """

    def __init__(
        self,
        confidence_levels: List[float] = [0.95, 0.99],
        lookback_days: int = 252,
        monte_carlo_sims: int = 10000
    ):
        self.confidence_levels = confidence_levels
        self.lookback_days = lookback_days
        self.monte_carlo_sims = monte_carlo_sims

    def historical_var(
        self,
        returns: pd.Series,
        portfolio_value: float,
        confidence: float = 0.95
    ) -> float:
        """Historical simulation VaR"""
        if len(returns) < 30:
            return portfolio_value * 0.02  # 2% default

        percentile = (1 - confidence) * 100
        var = -np.percentile(returns, percentile) * portfolio_value

        return var

    def parametric_var(
        self,
        returns: pd.Series,
        portfolio_value: float,
        confidence: float = 0.95
    ) -> float:
        """Parametric (Normal) VaR"""
        if len(returns) < 30:
            return portfolio_value * 0.02

        mu = returns.mean()
        sigma = returns.std()

        z_score = norm.ppf(1 - confidence)
        var = -(mu + z_score * sigma) * portfolio_value

        return var

    def cornish_fisher_var(
        self,
        returns: pd.Series,
        portfolio_value: float,
        confidence: float = 0.95
    ) -> float:
        """Cornish-Fisher VaR adjusting for skewness and kurtosis"""
        if len(returns) < 30:
            return portfolio_value * 0.02

        mu = returns.mean()
        sigma = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()

        z = norm.ppf(1 - confidence)

        # Cornish-Fisher expansion
        z_cf = (z + (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * (kurt - 3) / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)

        var = -(mu + z_cf * sigma) * portfolio_value

        return var

    def monte_carlo_var(
        self,
        returns: pd.Series,
        portfolio_value: float,
        confidence: float = 0.95,
        n_sims: int = None
    ) -> float:
        """Monte Carlo VaR simulation"""
        if len(returns) < 30:
            return portfolio_value * 0.02

        n_sims = n_sims or self.monte_carlo_sims
        mu = returns.mean()
        sigma = returns.std()

        # Generate random returns
        simulated_returns = np.random.normal(mu, sigma, n_sims)

        # Calculate VaR
        percentile = (1 - confidence) * 100
        var = -np.percentile(simulated_returns, percentile) * portfolio_value

        return var

    def calculate_cvar(
        self,
        returns: pd.Series,
        portfolio_value: float,
        confidence: float = 0.95
    ) -> float:
        """Conditional VaR (Expected Shortfall)"""
        if len(returns) < 30:
            return portfolio_value * 0.03

        percentile = (1 - confidence) * 100
        threshold = np.percentile(returns, percentile)

        # Average of returns below VaR threshold
        tail_returns = returns[returns <= threshold]
        cvar = -tail_returns.mean() * portfolio_value

        return cvar

    def calculate_all_metrics(
        self,
        returns: pd.Series,
        portfolio_value: float
    ) -> Dict[str, float]:
        """Calculate all VaR metrics"""
        metrics = {}

        for conf in self.confidence_levels:
            conf_str = str(int(conf * 100))

            metrics[f'var_{conf_str}_historical'] = self.historical_var(returns, portfolio_value, conf)
            metrics[f'var_{conf_str}_parametric'] = self.parametric_var(returns, portfolio_value, conf)
            metrics[f'var_{conf_str}_cornish_fisher'] = self.cornish_fisher_var(returns, portfolio_value, conf)
            metrics[f'cvar_{conf_str}'] = self.calculate_cvar(returns, portfolio_value, conf)

        return metrics


class RiskManager:
    """
    Institutional-grade risk management system.

    Features:
    - Real-time risk monitoring
    - Pre-trade risk validation
    - Position and concentration limits
    - VaR/CVaR monitoring
    - Circuit breakers
    - Stress testing
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        enable_circuit_breakers: bool = True,
        enable_auto_deleveraging: bool = True
    ):
        self.limits = limits or RiskLimits()
        self.enable_circuit_breakers = enable_circuit_breakers
        self.enable_auto_deleveraging = enable_auto_deleveraging

        self.var_calculator = VaRCalculator()

        # State tracking
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._portfolio_value: float = 0
        self._cash: float = 0
        self._peak_value: float = 0
        self._daily_pnl: float = 0
        self._weekly_pnl: float = 0
        self._monthly_pnl: float = 0
        self._daily_trades: int = 0
        self._daily_turnover: float = 0
        self._total_slippage: float = 0
        self._trade_count: int = 0

        # Historical data
        self._returns_history: pd.Series = pd.Series(dtype=float)
        self._pnl_history: List[Dict] = []
        self._risk_history: List[RiskMetrics] = []

        # Circuit breaker state
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_until: Optional[datetime] = None

        # Sector and correlation mappings
        self._sector_map: Dict[str, str] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None

        # Thread safety
        self._lock = threading.RLock()

    def set_sector_map(self, sector_map: Dict[str, str]) -> None:
        """Set symbol to sector mapping"""
        with self._lock:
            self._sector_map = sector_map

    def update_correlation_matrix(self, corr_matrix: pd.DataFrame) -> None:
        """Update correlation matrix"""
        with self._lock:
            self._correlation_matrix = corr_matrix

    def update_positions(
        self,
        positions: Dict[str, Dict[str, Any]],
        portfolio_value: float,
        cash: float
    ) -> None:
        """
        Update current positions.

        Args:
            positions: Dict of {symbol: {quantity, price, value, sector, ...}}
            portfolio_value: Total portfolio value
            cash: Available cash
        """
        with self._lock:
            self._positions = positions
            self._portfolio_value = portfolio_value
            self._cash = cash

            if portfolio_value > self._peak_value:
                self._peak_value = portfolio_value

    def update_pnl(
        self,
        daily_pnl: float,
        weekly_pnl: float = None,
        monthly_pnl: float = None
    ) -> None:
        """Update P&L metrics"""
        with self._lock:
            self._daily_pnl = daily_pnl
            if weekly_pnl is not None:
                self._weekly_pnl = weekly_pnl
            if monthly_pnl is not None:
                self._monthly_pnl = monthly_pnl

            # Check circuit breakers
            if self.enable_circuit_breakers:
                self._check_circuit_breakers()

    def update_returns(self, returns: pd.Series) -> None:
        """Update returns history"""
        with self._lock:
            self._returns_history = returns

    def record_trade(
        self,
        symbol: str,
        quantity: int,
        price: float,
        slippage_bps: float = 0
    ) -> None:
        """Record trade execution"""
        with self._lock:
            self._daily_trades += 1
            self._daily_turnover += abs(quantity * price)
            self._total_slippage += slippage_bps
            self._trade_count += 1

    def reset_daily_metrics(self) -> None:
        """Reset daily metrics (call at market open)"""
        with self._lock:
            self._daily_trades = 0
            self._daily_turnover = 0
            self._total_slippage = 0
            self._trade_count = 0

    def _check_circuit_breakers(self) -> None:
        """Check and activate circuit breakers if needed"""
        if not self.enable_circuit_breakers:
            return

        daily_loss_pct = -self._daily_pnl / self._portfolio_value if self._portfolio_value > 0 else 0

        if daily_loss_pct >= self.limits.circuit_breaker_loss:
            self._circuit_breaker_active = True
            self._circuit_breaker_until = datetime.now() + timedelta(
                minutes=self.limits.circuit_breaker_duration
            )

            audit_logger.log_risk_event(
                "CIRCUIT_BREAKER_TRIGGERED",
                "CRITICAL",
                {
                    'daily_loss_pct': daily_loss_pct,
                    'threshold': self.limits.circuit_breaker_loss,
                    'pause_until': self._circuit_breaker_until.isoformat()
                }
            )

            logger.critical(
                f"Circuit breaker triggered! Daily loss: {daily_loss_pct:.2%}. "
                f"Trading paused until {self._circuit_breaker_until}"
            )

    def is_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is currently allowed"""
        if self._circuit_breaker_active:
            if datetime.now() >= self._circuit_breaker_until:
                self._circuit_breaker_active = False
                self._circuit_breaker_until = None
                return True, "Circuit breaker reset"
            return False, f"Circuit breaker active until {self._circuit_breaker_until}"

        return True, "Trading allowed"

    def pre_trade_check(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        sector: Optional[str] = None
    ) -> PreTradeRiskCheck:
        """
        Perform comprehensive pre-trade risk checks.

        Returns:
            PreTradeRiskCheck with validation results
        """
        with self._lock:
            notional = quantity * price
            checks_passed = []
            checks_failed = []
            warnings = []

            # Circuit breaker check
            trading_allowed, reason = self.is_trading_allowed()
            if not trading_allowed:
                return PreTradeRiskCheck(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    notional=notional,
                    passed=False,
                    risk_score=100,
                    checks_failed=[f"Circuit breaker: {reason}"]
                )

            # 1. Position size check
            new_position_value = self._get_position_value(symbol) + notional * (1 if side == 'buy' else -1)
            position_pct = abs(new_position_value) / self._portfolio_value if self._portfolio_value > 0 else 0

            if position_pct > self.limits.max_position_pct:
                checks_failed.append(f"Position limit exceeded: {position_pct:.1%} > {self.limits.max_position_pct:.1%}")
            else:
                checks_passed.append("position_limit")

            # 2. Single name notional check
            if abs(new_position_value) > self.limits.max_single_name_notional:
                checks_failed.append(f"Single name notional exceeded: ${abs(new_position_value):,.0f}")
            else:
                checks_passed.append("single_name_notional")

            # 3. Sector concentration check
            sector = sector or self._sector_map.get(symbol, 'Other')
            sector_exposure = self._calculate_sector_exposure(sector)
            new_sector_exposure = sector_exposure + notional * (1 if side == 'buy' else -1)
            sector_pct = abs(new_sector_exposure) / self._portfolio_value if self._portfolio_value > 0 else 0

            if sector_pct > self.limits.max_sector_pct:
                checks_failed.append(f"Sector limit exceeded: {sector} at {sector_pct:.1%}")
            else:
                checks_passed.append("sector_limit")

            # 4. Gross exposure check
            gross_exposure = self._calculate_gross_exposure()
            new_gross = (gross_exposure + abs(notional)) / self._portfolio_value if self._portfolio_value > 0 else 0

            if new_gross > self.limits.max_gross_exposure:
                checks_failed.append(f"Gross exposure exceeded: {new_gross:.1%}")
            else:
                checks_passed.append("gross_exposure")

            # 5. Leverage check
            if new_gross > self.limits.max_leverage:
                warnings.append(f"Leverage elevated: {new_gross:.2f}x")

            # 6. Daily trade limit check
            if self._daily_trades >= self.limits.max_daily_trades:
                checks_failed.append(f"Daily trade limit reached: {self._daily_trades}")
            else:
                checks_passed.append("daily_trades")

            # 7. VaR impact check
            if len(self._returns_history) > 30:
                current_var = self.var_calculator.parametric_var(
                    self._returns_history,
                    self._portfolio_value,
                    0.95
                )
                var_pct = current_var / self._portfolio_value if self._portfolio_value > 0 else 0

                if var_pct > self.limits.max_var_pct:
                    warnings.append(f"VaR elevated: {var_pct:.2%}")

            # 8. Drawdown check
            drawdown = self._calculate_drawdown()
            if drawdown > self.limits.max_drawdown * 0.8:
                warnings.append(f"Approaching max drawdown: {drawdown:.2%}")

            if drawdown > self.limits.max_drawdown:
                checks_failed.append(f"Max drawdown breached: {drawdown:.2%}")

            # 9. Correlation check
            if self._correlation_matrix is not None:
                corr_exposure = self._calculate_correlation_exposure(symbol)
                if corr_exposure > self.limits.max_correlated_exposure:
                    warnings.append(f"High correlation exposure: {corr_exposure:.1%}")

            # Calculate risk score
            risk_score = self._calculate_risk_score(
                checks_passed, checks_failed, warnings
            )

            # Determine pass/fail
            passed = len(checks_failed) == 0

            # Calculate adjusted quantity if needed
            adjusted_quantity = None
            adjustment_reason = None

            if not passed and len(checks_failed) == 1 and "Position limit" in checks_failed[0]:
                # Calculate maximum allowed quantity
                max_position_value = self._portfolio_value * self.limits.max_position_pct
                current_position = self._get_position_value(symbol)
                available = max_position_value - abs(current_position)

                if available > 0:
                    adjusted_quantity = int(available / price)
                    if adjusted_quantity > 0:
                        adjustment_reason = "Reduced to position limit"
                        passed = True
                        checks_failed = []
                        checks_passed.append("position_limit_adjusted")

            # Log the check
            audit_logger.log_order(
                order_id=order_id,
                symbol=symbol,
                action="PRE_TRADE_CHECK",
                details={
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'passed': passed,
                    'risk_score': risk_score,
                    'checks_failed': checks_failed,
                    'warnings': warnings
                }
            )

            return PreTradeRiskCheck(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                notional=notional,
                passed=passed,
                risk_score=risk_score,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                warnings=warnings,
                adjusted_quantity=adjusted_quantity,
                adjustment_reason=adjustment_reason
            )

    def _get_position_value(self, symbol: str) -> float:
        """Get current position value for symbol"""
        if symbol in self._positions:
            return self._positions[symbol].get('value', 0)
        return 0

    def _calculate_sector_exposure(self, sector: str) -> float:
        """Calculate total exposure to a sector"""
        exposure = 0
        for symbol, pos in self._positions.items():
            pos_sector = pos.get('sector', self._sector_map.get(symbol, 'Other'))
            if pos_sector == sector:
                exposure += pos.get('value', 0)
        return exposure

    def _calculate_gross_exposure(self) -> float:
        """Calculate gross exposure"""
        return sum(abs(pos.get('value', 0)) for pos in self._positions.values())

    def _calculate_net_exposure(self) -> float:
        """Calculate net exposure"""
        return sum(pos.get('value', 0) for pos in self._positions.values())

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if self._peak_value <= 0:
            return 0
        return (self._peak_value - self._portfolio_value) / self._peak_value

    def _calculate_correlation_exposure(self, symbol: str) -> float:
        """Calculate exposure to correlated assets"""
        if self._correlation_matrix is None or symbol not in self._correlation_matrix.index:
            return 0

        correlated = []
        for other in self._correlation_matrix.columns:
            if other != symbol:
                corr = self._correlation_matrix.loc[symbol, other]
                if abs(corr) >= self.limits.correlation_threshold:
                    correlated.append(other)

        exposure = sum(
            abs(self._positions[s].get('value', 0))
            for s in correlated if s in self._positions
        )

        return exposure / self._portfolio_value if self._portfolio_value > 0 else 0

    def _calculate_risk_score(
        self,
        passed: List[str],
        failed: List[str],
        warnings: List[str]
    ) -> float:
        """Calculate overall risk score (0-100, higher = more risk)"""
        score = 0

        # Failed checks are severe
        score += len(failed) * 30

        # Warnings add moderate risk
        score += len(warnings) * 10

        # Cap at 100
        return min(score, 100)

    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive real-time risk metrics"""
        with self._lock:
            now = datetime.now()

            # Exposure calculations
            gross_exposure = self._calculate_gross_exposure()
            net_exposure = self._calculate_net_exposure()
            long_exposure = sum(
                pos.get('value', 0) for pos in self._positions.values()
                if pos.get('value', 0) > 0
            )
            short_exposure = abs(sum(
                pos.get('value', 0) for pos in self._positions.values()
                if pos.get('value', 0) < 0
            ))

            leverage = gross_exposure / self._portfolio_value if self._portfolio_value > 0 else 0

            # VaR calculations
            if len(self._returns_history) > 30:
                var_95 = self.var_calculator.parametric_var(
                    self._returns_history, self._portfolio_value, 0.95
                )
                var_99 = self.var_calculator.parametric_var(
                    self._returns_history, self._portfolio_value, 0.99
                )
                cvar_95 = self.var_calculator.calculate_cvar(
                    self._returns_history, self._portfolio_value, 0.95
                )
                cvar_99 = self.var_calculator.calculate_cvar(
                    self._returns_history, self._portfolio_value, 0.99
                )
                portfolio_volatility = self._returns_history.std() * np.sqrt(252)
            else:
                var_95 = self._portfolio_value * 0.02
                var_99 = self._portfolio_value * 0.03
                cvar_95 = self._portfolio_value * 0.03
                cvar_99 = self._portfolio_value * 0.045
                portfolio_volatility = 0.15

            # Drawdown
            current_drawdown = self._calculate_drawdown()
            max_drawdown = max(current_drawdown, getattr(self, '_max_drawdown_seen', 0))
            self._max_drawdown_seen = max_drawdown

            # Concentration
            if self._positions and self._portfolio_value > 0:
                position_weights = [
                    abs(pos.get('value', 0)) / self._portfolio_value
                    for pos in self._positions.values()
                ]
                largest_position_pct = max(position_weights) if position_weights else 0
                herfindahl_index = sum(w**2 for w in position_weights)

                # Sector concentration
                sector_exposures = defaultdict(float)
                for symbol, pos in self._positions.items():
                    sector = pos.get('sector', self._sector_map.get(symbol, 'Other'))
                    sector_exposures[sector] += abs(pos.get('value', 0))

                largest_sector_pct = max(
                    (v / self._portfolio_value for v in sector_exposures.values()),
                    default=0
                )
            else:
                largest_position_pct = 0
                largest_sector_pct = 0
                herfindahl_index = 0

            # Correlation risk
            if self._correlation_matrix is not None and len(self._positions) > 1:
                symbols = list(self._positions.keys())
                available = [s for s in symbols if s in self._correlation_matrix.index]
                if len(available) > 1:
                    sub_corr = self._correlation_matrix.loc[available, available]
                    avg_corr = (sub_corr.sum().sum() - len(available)) / (len(available) * (len(available) - 1))
                    correlation_risk = avg_corr
                else:
                    correlation_risk = 0
            else:
                correlation_risk = 0

            # Portfolio beta (simplified)
            portfolio_beta = 1.0  # Would need market returns for accurate calc

            # Trading metrics
            avg_slippage = self._total_slippage / self._trade_count if self._trade_count > 0 else 0

            # Risk alerts
            risk_alerts = []
            overall_risk_level = RiskLevel.LOW

            if current_drawdown > self.limits.max_drawdown:
                risk_alerts.append("DRAWDOWN_BREACH")
                overall_risk_level = RiskLevel.BREACH
            elif current_drawdown > self.limits.max_drawdown * 0.8:
                risk_alerts.append("DRAWDOWN_WARNING")
                overall_risk_level = max(overall_risk_level, RiskLevel.HIGH)

            if var_95 / self._portfolio_value > self.limits.max_var_pct:
                risk_alerts.append("VAR_BREACH")
                overall_risk_level = max(overall_risk_level, RiskLevel.HIGH)

            if leverage > self.limits.max_leverage:
                risk_alerts.append("LEVERAGE_BREACH")
                overall_risk_level = max(overall_risk_level, RiskLevel.CRITICAL)

            if portfolio_volatility > self.limits.max_volatility:
                risk_alerts.append("VOLATILITY_BREACH")
                overall_risk_level = max(overall_risk_level, RiskLevel.HIGH)

            if self._circuit_breaker_active:
                risk_alerts.append("CIRCUIT_BREAKER_ACTIVE")
                overall_risk_level = RiskLevel.BREACH

            metrics = RiskMetrics(
                timestamp=now,
                portfolio_value=self._portfolio_value,
                cash=self._cash,
                gross_exposure=gross_exposure,
                net_exposure=net_exposure,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                leverage=leverage,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                portfolio_volatility=portfolio_volatility,
                portfolio_beta=portfolio_beta,
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                daily_pnl=self._daily_pnl,
                weekly_pnl=self._weekly_pnl,
                monthly_pnl=self._monthly_pnl,
                largest_position_pct=largest_position_pct,
                largest_sector_pct=largest_sector_pct,
                herfindahl_index=herfindahl_index,
                correlation_risk=correlation_risk,
                daily_trades=self._daily_trades,
                daily_turnover=self._daily_turnover,
                avg_slippage_bps=avg_slippage,
                overall_risk_level=overall_risk_level,
                risk_alerts=risk_alerts
            )

            self._risk_history.append(metrics)

            # Keep last 1000 metrics
            if len(self._risk_history) > 1000:
                self._risk_history = self._risk_history[-1000:]

            return metrics

    def run_stress_test(
        self,
        scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Run stress test scenarios.

        Args:
            scenarios: Dict of {scenario_name: {symbol: shock_pct}}

        Returns:
            Dict of scenario results
        """
        results = {}

        for scenario_name, shocks in scenarios.items():
            scenario_pnl = 0
            scenario_details = {}

            for symbol, pos in self._positions.items():
                position_value = pos.get('value', 0)
                shock = shocks.get(symbol, shocks.get('market', 0))

                position_pnl = position_value * shock
                scenario_pnl += position_pnl
                scenario_details[symbol] = {
                    'position_value': position_value,
                    'shock': shock,
                    'pnl': position_pnl
                }

            results[scenario_name] = {
                'total_pnl': scenario_pnl,
                'pnl_pct': scenario_pnl / self._portfolio_value if self._portfolio_value > 0 else 0,
                'details': scenario_details
            }

        return results

    def get_deleveraging_trades(
        self,
        target_leverage: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Calculate trades needed to reduce leverage.

        Used for auto-deleveraging during high-risk periods.
        """
        if not self.enable_auto_deleveraging:
            return []

        current_leverage = self._calculate_gross_exposure() / self._portfolio_value if self._portfolio_value > 0 else 0

        if current_leverage <= target_leverage:
            return []

        scale = target_leverage / current_leverage
        trades = []

        for symbol, pos in self._positions.items():
            current_value = pos.get('value', 0)
            target_value = current_value * scale
            reduce_value = current_value - target_value

            if abs(reduce_value) > 1000:  # Min $1000 trade
                price = pos.get('price', 0)
                if price > 0:
                    shares = int(abs(reduce_value) / price)
                    if shares > 0:
                        trades.append({
                            'symbol': symbol,
                            'side': 'sell' if current_value > 0 else 'buy',
                            'shares': shares,
                            'reason': 'deleveraging'
                        })

        return trades


class BacktestCircuitBreaker:
    """
    Enhanced circuit breaker for backtesting with multiple trigger conditions.

    ADDED: This class addresses the audit finding that circuit breakers
    were not properly integrated into the backtest loop.

    Trigger Conditions:
    1. Intraday loss exceeds threshold (e.g., -3%)
    2. Drawdown exceeds threshold (e.g., -15%)
    3. Consecutive losing trades exceed threshold (e.g., 5)
    4. Volatility spike detected (e.g., 3x normal)
    5. Win rate drops below threshold (e.g., <40% over last N trades)

    When triggered:
    - Pauses new position entries
    - Can optionally force-close existing positions
    - Cooldown period before resuming
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,        # 3% daily loss trigger
        max_drawdown_pct: float = 0.15,          # 15% drawdown trigger
        max_consecutive_losses: int = 5,          # 5 consecutive losses
        volatility_spike_mult: float = 3.0,       # 3x normal volatility
        min_win_rate: float = 0.35,               # 35% minimum win rate
        win_rate_lookback: int = 20,              # Over last 20 trades
        cooldown_periods: int = 10,               # Pause for 10 bars
        force_close_on_trigger: bool = False      # Close positions when triggered
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.volatility_spike_mult = volatility_spike_mult
        self.min_win_rate = min_win_rate
        self.win_rate_lookback = win_rate_lookback
        self.cooldown_periods = cooldown_periods
        self.force_close_on_trigger = force_close_on_trigger

        # State
        self._is_triggered = False
        self._trigger_reason = None
        self._trigger_time = None
        self._cooldown_remaining = 0
        self._consecutive_losses = 0
        self._trade_results: List[bool] = []  # True = win, False = loss
        self._normal_volatility = None
        self._daily_starting_equity = None
        self._peak_equity = 0

        # Metrics
        self._triggers_count = 0
        self._total_cooldown_periods = 0

    def reset_daily(self, current_equity: float) -> None:
        """Reset daily metrics (call at start of each trading day)."""
        self._daily_starting_equity = current_equity

    def update_peak(self, current_equity: float) -> None:
        """Update peak equity for drawdown calculation."""
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

    def set_normal_volatility(self, volatility: float) -> None:
        """Set baseline volatility for spike detection."""
        self._normal_volatility = volatility

    def record_trade(self, pnl: float) -> None:
        """Record trade result for consecutive loss and win rate tracking."""
        is_win = pnl > 0
        self._trade_results.append(is_win)

        # Track consecutive losses
        if not is_win:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Keep only recent trades for win rate
        if len(self._trade_results) > self.win_rate_lookback * 2:
            self._trade_results = self._trade_results[-self.win_rate_lookback:]

    def check_triggers(
        self,
        current_equity: float,
        current_volatility: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check all circuit breaker conditions.

        Args:
            current_equity: Current portfolio equity
            current_volatility: Current realized volatility (optional)

        Returns:
            Tuple of (is_triggered, trigger_reason)
        """
        # If in cooldown, decrement and check if done
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                self._is_triggered = False
                self._trigger_reason = None
                logger.info("Circuit breaker cooldown complete, trading resumed")
            return self._is_triggered, self._trigger_reason

        # Update peak
        self.update_peak(current_equity)

        # 1. Check daily loss
        if self._daily_starting_equity is not None and self._daily_starting_equity > 0:
            daily_return = (current_equity - self._daily_starting_equity) / self._daily_starting_equity
            if daily_return < -self.max_daily_loss_pct:
                return self._trigger(f"Daily loss exceeded: {daily_return:.2%}")

        # 2. Check drawdown
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown > self.max_drawdown_pct:
                return self._trigger(f"Max drawdown exceeded: {drawdown:.2%}")

        # 3. Check consecutive losses
        if self._consecutive_losses >= self.max_consecutive_losses:
            return self._trigger(f"Consecutive losses: {self._consecutive_losses}")

        # 4. Check volatility spike
        if current_volatility is not None and self._normal_volatility is not None:
            if current_volatility > self._normal_volatility * self.volatility_spike_mult:
                return self._trigger(
                    f"Volatility spike: {current_volatility:.2%} vs normal {self._normal_volatility:.2%}"
                )

        # 5. Check win rate
        if len(self._trade_results) >= self.win_rate_lookback:
            recent_results = self._trade_results[-self.win_rate_lookback:]
            win_rate = sum(recent_results) / len(recent_results)
            if win_rate < self.min_win_rate:
                return self._trigger(f"Win rate collapsed: {win_rate:.2%}")

        return False, None

    def _trigger(self, reason: str) -> Tuple[bool, str]:
        """Trigger the circuit breaker."""
        self._is_triggered = True
        self._trigger_reason = reason
        self._trigger_time = datetime.now()
        self._cooldown_remaining = self.cooldown_periods
        self._triggers_count += 1
        self._total_cooldown_periods += self.cooldown_periods

        logger.warning(f"CIRCUIT BREAKER TRIGGERED: {reason}")
        logger.warning(f"Trading paused for {self.cooldown_periods} periods")

        return True, reason

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return not self._is_triggered

    def should_force_close(self) -> bool:
        """Check if positions should be force-closed."""
        return self._is_triggered and self.force_close_on_trigger

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        recent_win_rate = None
        if len(self._trade_results) >= 5:
            recent = self._trade_results[-min(self.win_rate_lookback, len(self._trade_results)):]
            recent_win_rate = sum(recent) / len(recent)

        return {
            'is_triggered': self._is_triggered,
            'trigger_reason': self._trigger_reason,
            'cooldown_remaining': self._cooldown_remaining,
            'consecutive_losses': self._consecutive_losses,
            'recent_win_rate': recent_win_rate,
            'triggers_count': self._triggers_count,
            'total_cooldown_periods': self._total_cooldown_periods
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics for reporting."""
        return {
            'total_triggers': self._triggers_count,
            'total_cooldown_periods': self._total_cooldown_periods,
            'consecutive_losses_max': self._consecutive_losses,
            'trades_tracked': len(self._trade_results)
        }
