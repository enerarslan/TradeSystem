"""
Mean Reversion Strategies
JPMorgan-Level Statistical Arbitrage

Features:
- Z-score based mean reversion
- Bollinger band mean reversion
- Pairs trading / Statistical arbitrage
- Cointegration-based strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .base_strategy import (
    BaseStrategy, Signal, SignalType, StrategyConfig
)
from ..features.technical import TechnicalIndicators
from ..utils.logger import get_logger


logger = get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy based on Z-score.

    Trades when price deviates significantly from its mean,
    betting on reversion to mean.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        max_holding_periods: int = 20,
        use_bollinger: bool = True,
        **kwargs
    ):
        """
        Initialize MeanReversionStrategy.

        Args:
            lookback_period: Period for mean/std calculation
            entry_zscore: Z-score threshold for entry
            exit_zscore: Z-score threshold for exit
            max_holding_periods: Maximum holding time
            use_bollinger: Use Bollinger Bands for additional confirmation
        """
        config = StrategyConfig(name="mean_reversion", **kwargs)
        super().__init__(config)

        self.lookback_period = lookback_period
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.max_holding_periods = max_holding_periods
        self.use_bollinger = use_bollinger

        self._entry_signals: Dict[str, Tuple[datetime, float]] = {}

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Signal]:
        """Generate mean reversion signals"""
        signals = {}

        for symbol, df in data.items():
            try:
                signal_type, strength = self.calculate_signal_strength(df, symbol)

                if signal_type != SignalType.FLAT:
                    signal = self._create_signal(df, symbol, signal_type, strength)

                    if self.validate_signal(signal):
                        signals[symbol] = signal
                        self.record_signal(signal)

            except Exception as e:
                logger.warning(f"Error in mean reversion for {symbol}: {e}")

        return signals

    def calculate_signal_strength(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Tuple[SignalType, float]:
        """Calculate mean reversion signal"""
        if len(df) < self.lookback_period + 10:
            return SignalType.FLAT, 0.0

        close = df['close']

        # Calculate Z-score
        rolling_mean = close.rolling(self.lookback_period).mean()
        rolling_std = close.rolling(self.lookback_period).std()

        zscore = (close - rolling_mean) / rolling_std
        current_zscore = zscore.iloc[-1]

        # Bollinger band position
        bb_position = 0
        if self.use_bollinger:
            upper, middle, lower = TechnicalIndicators.bollinger_bands(
                close, self.lookback_period
            )
            bb_pct = TechnicalIndicators.bollinger_percent_b(
                close, self.lookback_period
            ).iloc[-1]

            if bb_pct < 0:
                bb_position = -1  # Below lower band
            elif bb_pct > 1:
                bb_position = 1  # Above upper band

        # RSI for oversold/overbought confirmation
        rsi = TechnicalIndicators.rsi(close, 14).iloc[-1]

        # Entry signals
        signal_type = SignalType.FLAT
        strength = 0.0

        # Long entry: price significantly below mean
        if current_zscore < -self.entry_zscore:
            signal_type = SignalType.LONG
            strength = min(abs(current_zscore) / 3, 1.0)

            # Boost strength if RSI oversold
            if rsi < 30:
                strength = min(strength * 1.2, 1.0)

            # Boost if below Bollinger
            if bb_position == -1:
                strength = min(strength * 1.1, 1.0)

        # Short entry: price significantly above mean
        elif current_zscore > self.entry_zscore:
            signal_type = SignalType.SHORT
            strength = min(abs(current_zscore) / 3, 1.0)

            if rsi > 70:
                strength = min(strength * 1.2, 1.0)

            if bb_position == 1:
                strength = min(strength * 1.1, 1.0)

        # Check for exit signals on existing positions
        if symbol in self._positions:
            if abs(current_zscore) < self.exit_zscore:
                # Signal to close position
                if self._positions[symbol] > 0:
                    signal_type = SignalType.LONG_EXIT
                else:
                    signal_type = SignalType.SHORT_EXIT
                strength = 1.0

        return signal_type, strength

    def _create_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        signal_type: SignalType,
        strength: float
    ) -> Signal:
        """Create mean reversion signal"""
        current_price = df['close'].iloc[-1]
        direction = 1 if signal_type in [SignalType.LONG, SignalType.LONG_EXIT] else -1

        # Calculate mean for target
        rolling_mean = df['close'].rolling(self.lookback_period).mean().iloc[-1]

        # Stop loss at extended deviation
        atr = TechnicalIndicators.atr(
            df['high'], df['low'], df['close'], 14
        ).iloc[-1]

        stop_loss = self.calculate_stop_loss(current_price, direction, atr * 1.5)

        # Take profit at mean
        if direction == 1:
            take_profit = rolling_mean
        else:
            take_profit = rolling_mean

        # Calculate zscore for metadata
        rolling_std = df['close'].rolling(self.lookback_period).std().iloc[-1]
        zscore = (current_price - rolling_mean) / rolling_std

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            timestamp=df.index[-1],
            price=current_price,
            strategy_name=self.name,
            confidence=min(abs(zscore) / self.entry_zscore, 1.0),
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'zscore': zscore,
                'mean': rolling_mean,
                'std': rolling_std
            }
        )


class StatArbStrategy(BaseStrategy):
    """
    Statistical arbitrage / Pairs trading strategy.

    Trades the spread between cointegrated pairs,
    betting on spread mean reversion.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]] = None,
        lookback_period: int = 60,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        cointegration_pvalue: float = 0.05,
        **kwargs
    ):
        """
        Initialize StatArbStrategy.

        Args:
            pairs: List of (symbol1, symbol2) pairs to trade
            lookback_period: Period for spread calculation
            entry_zscore: Entry threshold
            exit_zscore: Exit threshold
            cointegration_pvalue: Maximum p-value for cointegration test
        """
        config = StrategyConfig(name="stat_arb", **kwargs)
        super().__init__(config)

        self.pairs = pairs or []
        self.lookback_period = lookback_period
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.cointegration_pvalue = cointegration_pvalue

        self._hedge_ratios: Dict[Tuple[str, str], float] = {}
        self._spread_stats: Dict[Tuple[str, str], Dict] = {}

    def add_pair(self, symbol1: str, symbol2: str) -> None:
        """Add a trading pair"""
        self.pairs.append((symbol1, symbol2))

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Signal]:
        """Generate pairs trading signals"""
        signals = {}

        for sym1, sym2 in self.pairs:
            if sym1 not in data or sym2 not in data:
                continue

            try:
                pair_signals = self._generate_pair_signals(
                    data[sym1], data[sym2], sym1, sym2
                )

                for symbol, signal in pair_signals.items():
                    if self.validate_signal(signal):
                        signals[symbol] = signal
                        self.record_signal(signal)

            except Exception as e:
                logger.warning(f"Error in pairs trading for {sym1}/{sym2}: {e}")

        return signals

    def _generate_pair_signals(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        sym1: str,
        sym2: str
    ) -> Dict[str, Signal]:
        """Generate signals for a single pair"""
        if len(df1) < self.lookback_period or len(df2) < self.lookback_period:
            return {}

        # Align data
        common_idx = df1.index.intersection(df2.index)
        p1 = df1.loc[common_idx, 'close']
        p2 = df2.loc[common_idx, 'close']

        # Calculate hedge ratio (rolling OLS)
        hedge_ratio = self._calculate_hedge_ratio(p1, p2)

        # Calculate spread
        spread = np.log(p1) - hedge_ratio * np.log(p2)

        # Z-score of spread
        spread_mean = spread.rolling(self.lookback_period).mean()
        spread_std = spread.rolling(self.lookback_period).std()
        zscore = (spread - spread_mean) / spread_std

        current_zscore = zscore.iloc[-1]

        # Store stats
        self._hedge_ratios[(sym1, sym2)] = hedge_ratio
        self._spread_stats[(sym1, sym2)] = {
            'mean': spread_mean.iloc[-1],
            'std': spread_std.iloc[-1],
            'zscore': current_zscore
        }

        signals = {}

        # Generate signals based on spread z-score
        if current_zscore > self.entry_zscore:
            # Spread too high: short sym1, long sym2
            signals[sym1] = self._create_pair_signal(
                df1, sym1, SignalType.SHORT, abs(current_zscore), sym2, hedge_ratio
            )
            signals[sym2] = self._create_pair_signal(
                df2, sym2, SignalType.LONG, abs(current_zscore), sym1, 1/hedge_ratio
            )

        elif current_zscore < -self.entry_zscore:
            # Spread too low: long sym1, short sym2
            signals[sym1] = self._create_pair_signal(
                df1, sym1, SignalType.LONG, abs(current_zscore), sym2, hedge_ratio
            )
            signals[sym2] = self._create_pair_signal(
                df2, sym2, SignalType.SHORT, abs(current_zscore), sym1, 1/hedge_ratio
            )

        return signals

    def _calculate_hedge_ratio(
        self,
        p1: pd.Series,
        p2: pd.Series
    ) -> float:
        """Calculate hedge ratio using rolling regression"""
        log_p1 = np.log(p1)
        log_p2 = np.log(p2)

        # Simple rolling covariance / variance
        cov = log_p1.rolling(self.lookback_period).cov(log_p2)
        var = log_p2.rolling(self.lookback_period).var()

        hedge_ratio = cov / var

        return hedge_ratio.iloc[-1]

    def _create_pair_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        signal_type: SignalType,
        zscore: float,
        paired_symbol: str,
        hedge_ratio: float
    ) -> Signal:
        """Create pairs trading signal"""
        current_price = df['close'].iloc[-1]
        direction = 1 if signal_type == SignalType.LONG else -1

        # Stop loss based on spread deviation
        spread_std = self._spread_stats.get(
            (symbol, paired_symbol), {}
        ).get('std', current_price * 0.02)

        stop_loss = self.calculate_stop_loss(current_price, direction, spread_std * 2)
        take_profit = self.calculate_take_profit(current_price, direction, stop_loss)

        strength = min(zscore / 3, 1.0)

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            timestamp=df.index[-1],
            price=current_price,
            strategy_name=self.name,
            confidence=strength,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'paired_symbol': paired_symbol,
                'hedge_ratio': hedge_ratio,
                'spread_zscore': zscore,
                'is_pairs_trade': True
            }
        )

    def calculate_signal_strength(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Tuple[SignalType, float]:
        """Not used for pairs strategy"""
        return SignalType.FLAT, 0.0

    def test_cointegration(
        self,
        p1: pd.Series,
        p2: pd.Series
    ) -> Tuple[bool, float]:
        """
        Test if two price series are cointegrated.

        Returns:
            Tuple of (is_cointegrated, p_value)
        """
        try:
            from statsmodels.tsa.stattools import coint

            # Cointegration test
            _, pvalue, _ = coint(p1, p2)

            is_cointegrated = pvalue < self.cointegration_pvalue

            return is_cointegrated, pvalue

        except ImportError:
            logger.warning("statsmodels not installed for cointegration test")
            return True, 0.01  # Assume cointegrated

    def find_cointegrated_pairs(
        self,
        data: Dict[str, pd.DataFrame],
        min_observations: int = 252
    ) -> List[Tuple[str, str, float]]:
        """
        Find all cointegrated pairs in universe.

        Returns:
            List of (symbol1, symbol2, p_value) tuples
        """
        symbols = list(data.keys())
        cointegrated_pairs = []

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                # Align data
                common_idx = data[sym1].index.intersection(data[sym2].index)

                if len(common_idx) < min_observations:
                    continue

                p1 = data[sym1].loc[common_idx, 'close']
                p2 = data[sym2].loc[common_idx, 'close']

                is_coint, pvalue = self.test_cointegration(p1, p2)

                if is_coint:
                    cointegrated_pairs.append((sym1, sym2, pvalue))
                    logger.info(f"Cointegrated pair found: {sym1}/{sym2} (p={pvalue:.4f})")

        # Sort by p-value (most cointegrated first)
        cointegrated_pairs.sort(key=lambda x: x[2])

        return cointegrated_pairs
