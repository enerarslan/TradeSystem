"""
Market Microstructure Features
JPMorgan-Level Order Flow and Microstructure Analysis

Features:
- Order flow imbalance
- Trade intensity
- Price impact models
- Bid-ask spread analysis
- Kyle's Lambda estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..utils.logger import get_logger
from ..utils.helpers import safe_divide


logger = get_logger(__name__)


@dataclass
class MicrostructureMetrics:
    """Container for microstructure metrics"""
    kyle_lambda: float
    amihud_illiquidity: float
    roll_spread: float
    effective_spread: float
    price_impact: float
    order_flow_toxicity: float


class MicrostructureFeatures:
    """
    Market microstructure feature generator.

    Generates features related to:
    - Liquidity estimation
    - Order flow analysis
    - Price impact
    - Information asymmetry
    """

    @staticmethod
    def estimate_bid_ask_spread(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        method: str = 'roll'
    ) -> pd.Series:
        """
        Estimate bid-ask spread from OHLC data.

        Methods:
        - roll: Roll (1984) spread estimator
        - parkinson: Based on high-low range
        - corwin_schultz: Corwin-Schultz (2012) high-low spread
        """
        if method == 'roll':
            return MicrostructureFeatures._roll_spread(close)
        elif method == 'parkinson':
            return MicrostructureFeatures._parkinson_spread(high, low)
        elif method == 'corwin_schultz':
            return MicrostructureFeatures._corwin_schultz_spread(high, low)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _roll_spread(close: pd.Series, window: int = 20) -> pd.Series:
        """
        Roll (1984) spread estimator.

        Spread = 2 * sqrt(-Cov(r_t, r_{t-1}))
        """
        returns = close.pct_change()
        cov = returns.rolling(window).apply(
            lambda x: np.cov(x[1:], x[:-1])[0, 1] if len(x) > 1 else 0,
            raw=True
        )

        # Spread is only valid when covariance is negative
        spread = 2 * np.sqrt(np.maximum(-cov, 0))

        return spread

    @staticmethod
    def _parkinson_spread(high: pd.Series, low: pd.Series) -> pd.Series:
        """Parkinson-based spread estimator"""
        log_hl = np.log(high / low)
        return log_hl / (2 * np.sqrt(np.log(2)))

    @staticmethod
    def _corwin_schultz_spread(
        high: pd.Series,
        low: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Corwin-Schultz (2012) high-low spread estimator.

        More accurate than Roll spread for daily data.
        """
        beta = np.log(high / low) ** 2
        gamma = np.log(
            high.rolling(2).max() / low.rolling(2).min()
        ) ** 2

        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / \
                (3 - 2 * np.sqrt(2)) - \
                np.sqrt(gamma / (3 - 2 * np.sqrt(2)))

        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

        # Smooth with rolling window
        spread = spread.rolling(window).mean()

        # Replace negatives with zero
        spread = spread.clip(lower=0)

        return spread

    @staticmethod
    def amihud_illiquidity(
        close: pd.Series,
        volume: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Amihud (2002) illiquidity measure.

        ILLIQ = |Return| / Dollar Volume
        """
        returns = close.pct_change().abs()
        dollar_volume = close * volume

        illiq = safe_divide(returns, dollar_volume)

        # Scale and smooth
        illiq = illiq.rolling(window).mean() * 1e6

        return illiq

    @staticmethod
    def kyle_lambda(
        close: pd.Series,
        volume: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Kyle's Lambda - price impact coefficient.

        Estimated via regression: ΔP = λ * SignedVolume + ε
        """
        returns = close.pct_change()
        signed_volume = np.sign(returns) * volume

        def estimate_lambda(data):
            if len(data) < 10:
                return 0

            returns_window = data['returns']
            sv_window = data['signed_volume']

            # Simple OLS: λ = Cov(r, sv) / Var(sv)
            cov = np.cov(returns_window, sv_window)[0, 1]
            var = np.var(sv_window)

            return safe_divide(cov, var)

        # Combine into DataFrame for rolling
        df = pd.DataFrame({
            'returns': returns,
            'signed_volume': signed_volume
        })

        kyle_lambda = df.rolling(window).apply(
            lambda x: estimate_lambda(pd.DataFrame({
                'returns': x[:len(x)//2],
                'signed_volume': x[len(x)//2:]
            })) if len(x) > 10 else 0,
            raw=True
        )

        # Simplified approach
        kyle_lambda = safe_divide(
            returns.rolling(window).cov(signed_volume),
            signed_volume.rolling(window).var()
        )

        return kyle_lambda

    @staticmethod
    def vpin(
        close: pd.Series,
        volume: pd.Series,
        bucket_size: int = 50,
        n_buckets: int = 50
    ) -> pd.Series:
        """
        Volume-Synchronized Probability of Informed Trading (VPIN).

        Measures order flow toxicity.
        """
        returns = close.pct_change()

        # Classify trades as buy or sell
        buy_volume = volume.where(returns > 0, 0)
        sell_volume = volume.where(returns < 0, 0)

        # Calculate order imbalance
        imbalance = abs(buy_volume - sell_volume)

        # VPIN = Average |Imbalance| / Total Volume
        vpin = imbalance.rolling(n_buckets).sum() / volume.rolling(n_buckets).sum()

        return vpin

    @staticmethod
    def order_flow_imbalance(
        close: pd.Series,
        volume: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Order Flow Imbalance (OFI).

        Measures the net buying/selling pressure.
        """
        returns = close.pct_change()

        # Classify volume
        buy_volume = volume.where(returns > 0, 0)
        sell_volume = volume.where(returns < 0, 0)

        # OFI = (Buy Volume - Sell Volume) / Total Volume
        ofi = (buy_volume - sell_volume).rolling(window).sum() / \
              volume.rolling(window).sum()

        return ofi

    @staticmethod
    def trade_intensity(
        volume: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Trade intensity - normalized volume activity.
        """
        vol_ma = volume.rolling(window * 5).mean()
        intensity = volume / vol_ma

        return intensity

    @staticmethod
    def realized_variance(
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Realized variance from high-frequency returns.
        """
        returns = close.pct_change()
        rv = (returns ** 2).rolling(window).sum()

        return rv

    @staticmethod
    def realized_bipower_variation(
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Realized bipower variation - robust to jumps.
        """
        returns = close.pct_change().abs()
        bv = returns * returns.shift(1)
        rbv = bv.rolling(window).sum() * (np.pi / 2)

        return rbv

    @staticmethod
    def jump_detection(
        close: pd.Series,
        window: int = 20,
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect price jumps using realized variance vs bipower variation.

        Returns 1 if jump detected, 0 otherwise.
        """
        rv = MicrostructureFeatures.realized_variance(close, window)
        rbv = MicrostructureFeatures.realized_bipower_variation(close, window)

        # Jump ratio
        ratio = safe_divide(rv, rbv)

        # Detect jumps
        jumps = (ratio > threshold).astype(int)

        return jumps

    @staticmethod
    def price_impact_model(
        close: pd.Series,
        volume: pd.Series,
        window: int = 50
    ) -> Dict[str, pd.Series]:
        """
        Estimate price impact model parameters.

        Returns temporary and permanent impact estimates.
        """
        returns = close.pct_change()
        signed_volume = np.sign(returns) * np.sqrt(volume)  # Square-root model

        # Temporary impact (same-period)
        temp_impact = safe_divide(
            returns.rolling(window).cov(signed_volume),
            signed_volume.rolling(window).var()
        )

        # Permanent impact (lagged)
        lagged_sv = signed_volume.shift(1)
        perm_impact = safe_divide(
            returns.rolling(window).cov(lagged_sv),
            lagged_sv.rolling(window).var()
        )

        return {
            'temporary_impact': temp_impact,
            'permanent_impact': perm_impact,
            'total_impact': temp_impact + perm_impact
        }

    @staticmethod
    def generate_all_features(
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Generate all microstructure features.

        Args:
            df: OHLCV DataFrame
            window: Lookback window

        Returns:
            DataFrame with all microstructure features
        """
        features = pd.DataFrame(index=df.index)

        # Spread estimates
        features['roll_spread'] = MicrostructureFeatures._roll_spread(
            df['close'], window
        )
        features['cs_spread'] = MicrostructureFeatures._corwin_schultz_spread(
            df['high'], df['low'], window
        )

        # Liquidity measures
        features['amihud_illiq'] = MicrostructureFeatures.amihud_illiquidity(
            df['close'], df['volume'], window
        )
        features['kyle_lambda'] = MicrostructureFeatures.kyle_lambda(
            df['close'], df['volume'], window
        )

        # Order flow
        features['vpin'] = MicrostructureFeatures.vpin(
            df['close'], df['volume']
        )
        features['ofi'] = MicrostructureFeatures.order_flow_imbalance(
            df['close'], df['volume'], window
        )
        features['trade_intensity'] = MicrostructureFeatures.trade_intensity(
            df['volume'], window
        )

        # Volatility measures
        features['realized_var'] = MicrostructureFeatures.realized_variance(
            df['close'], window
        )
        features['bipower_var'] = MicrostructureFeatures.realized_bipower_variation(
            df['close'], window
        )
        features['jump_indicator'] = MicrostructureFeatures.jump_detection(
            df['close'], window
        )

        # Price impact
        impact = MicrostructureFeatures.price_impact_model(
            df['close'], df['volume'], window
        )
        features['temp_impact'] = impact['temporary_impact']
        features['perm_impact'] = impact['permanent_impact']

        return features
