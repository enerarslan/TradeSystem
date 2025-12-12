"""
Market Microstructure Features
JPMorgan-Level Order Flow and Microstructure Analysis

Features:
- Order flow imbalance
- Trade intensity
- Price impact models
- Bid-ask spread analysis
- Kyle's Lambda estimation
- Level 2 Order Book Features (NEW)
  - Order Book Imbalance (OBI)
  - Weighted Mid-Price (Micro-Price)
  - Bid-Ask Spread Volatility
  - Effective Spread
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

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


@dataclass
class OrderBookLevel:
    """Single level in the order book"""
    price: float
    size: float
    order_count: int = 1


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book at a point in time"""
    timestamp: pd.Timestamp
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def best_bid_size(self) -> Optional[float]:
        return self.bids[0].size if self.bids else None

    @property
    def best_ask_size(self) -> Optional[float]:
        return self.asks[0].size if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


@dataclass
class Level2Metrics:
    """Container for Level 2 microstructure metrics"""
    obi: float  # Order Book Imbalance
    micro_price: float  # Volume-weighted mid-price
    spread: float  # Bid-ask spread
    spread_bps: float  # Spread in basis points
    spread_volatility: float  # Rolling spread volatility
    effective_spread: float  # Effective spread
    depth_imbalance: float  # Depth imbalance across levels
    total_bid_depth: float  # Total bid side depth
    total_ask_depth: float  # Total ask side depth


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


class Level2Features:
    """
    Level 2 (Order Book) Microstructure Features.

    Calculates features from order book data including:
    - Order Book Imbalance (OBI) at top N levels
    - Weighted Mid-Price (Micro-Price) using volume-weighted bid/ask
    - Bid-Ask Spread Volatility
    - Effective Spread
    """

    @staticmethod
    def order_book_imbalance(
        bid_sizes: Union[pd.Series, np.ndarray],
        ask_sizes: Union[pd.Series, np.ndarray],
        levels: int = 1
    ) -> Union[float, pd.Series]:
        """
        Calculate Order Book Imbalance (OBI) ratio.

        OBI = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)

        Range: [-1, 1]
        - Positive: More buying pressure
        - Negative: More selling pressure

        Args:
            bid_sizes: Bid sizes (single value or series for top N levels)
            ask_sizes: Ask sizes (single value or series for top N levels)
            levels: Number of levels to consider

        Returns:
            OBI value or series
        """
        bid_total = np.sum(bid_sizes) if hasattr(bid_sizes, '__len__') else bid_sizes
        ask_total = np.sum(ask_sizes) if hasattr(ask_sizes, '__len__') else ask_sizes

        return safe_divide(bid_total - ask_total, bid_total + ask_total)

    @staticmethod
    def order_book_imbalance_multi_level(
        bid_prices: np.ndarray,
        bid_sizes: np.ndarray,
        ask_prices: np.ndarray,
        ask_sizes: np.ndarray,
        n_levels: int = 5,
        decay: float = 0.5
    ) -> float:
        """
        Calculate OBI with exponential decay weighting by level.

        Closer levels get higher weights.

        Args:
            bid_prices: Array of bid prices (best to worst)
            bid_sizes: Array of bid sizes
            ask_prices: Array of ask prices (best to worst)
            ask_sizes: Array of ask sizes
            n_levels: Number of levels to consider
            decay: Decay factor for level weighting

        Returns:
            Weighted OBI
        """
        n_levels = min(n_levels, len(bid_sizes), len(ask_sizes))

        if n_levels == 0:
            return 0.0

        # Exponential decay weights
        weights = np.exp(-decay * np.arange(n_levels))
        weights = weights / weights.sum()

        # Weighted bid and ask volumes
        bid_weighted = np.sum(bid_sizes[:n_levels] * weights)
        ask_weighted = np.sum(ask_sizes[:n_levels] * weights)

        return safe_divide(bid_weighted - ask_weighted, bid_weighted + ask_weighted)

    @staticmethod
    def weighted_mid_price(
        best_bid: float,
        best_ask: float,
        bid_size: float,
        ask_size: float
    ) -> float:
        """
        Calculate Weighted Mid-Price (Micro-Price).

        This is a volume-weighted average that adjusts the mid-price
        based on order book imbalance. More accurate than simple mid-price.

        Micro-Price = Bid * (Ask_Vol / Total_Vol) + Ask * (Bid_Vol / Total_Vol)

        When bid_size > ask_size, price is pushed toward ask (buying pressure)
        When ask_size > bid_size, price is pushed toward bid (selling pressure)

        Args:
            best_bid: Best bid price
            best_ask: Best ask price
            bid_size: Best bid size
            ask_size: Best ask size

        Returns:
            Micro-price
        """
        total_size = bid_size + ask_size

        if total_size == 0:
            return (best_bid + best_ask) / 2

        # Volume-weighted mid-price (micro-price)
        # When there's more bid volume, price is pushed toward ask
        micro_price = (best_bid * ask_size + best_ask * bid_size) / total_size

        return micro_price

    @staticmethod
    def weighted_mid_price_multi_level(
        bid_prices: np.ndarray,
        bid_sizes: np.ndarray,
        ask_prices: np.ndarray,
        ask_sizes: np.ndarray,
        n_levels: int = 5
    ) -> float:
        """
        Calculate weighted mid-price using multiple levels.

        Args:
            bid_prices: Array of bid prices
            bid_sizes: Array of bid sizes
            ask_prices: Array of ask prices
            ask_sizes: Array of ask sizes
            n_levels: Number of levels to use

        Returns:
            Multi-level weighted mid-price
        """
        n_levels = min(n_levels, len(bid_sizes), len(ask_sizes))

        if n_levels == 0:
            return np.nan

        bid_vwap = np.average(bid_prices[:n_levels], weights=bid_sizes[:n_levels])
        ask_vwap = np.average(ask_prices[:n_levels], weights=ask_sizes[:n_levels])

        total_bid = np.sum(bid_sizes[:n_levels])
        total_ask = np.sum(ask_sizes[:n_levels])
        total = total_bid + total_ask

        if total == 0:
            return (bid_vwap + ask_vwap) / 2

        return (bid_vwap * total_ask + ask_vwap * total_bid) / total

    @staticmethod
    def bid_ask_spread(
        best_bid: float,
        best_ask: float,
        in_bps: bool = False
    ) -> float:
        """
        Calculate bid-ask spread.

        Args:
            best_bid: Best bid price
            best_ask: Best ask price
            in_bps: Return spread in basis points

        Returns:
            Spread (absolute or in bps)
        """
        spread = best_ask - best_bid

        if in_bps:
            mid = (best_bid + best_ask) / 2
            return (spread / mid) * 10000 if mid > 0 else 0

        return spread

    @staticmethod
    def bid_ask_spread_volatility(
        spreads: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate rolling bid-ask spread volatility.

        High spread volatility indicates market stress or illiquidity.

        Args:
            spreads: Series of bid-ask spreads
            window: Rolling window size

        Returns:
            Rolling spread volatility
        """
        return spreads.rolling(window).std()

    @staticmethod
    def effective_spread(
        trade_price: float,
        mid_price: float,
        trade_direction: int = 1
    ) -> float:
        """
        Calculate effective spread.

        Measures actual cost of executing a trade.

        Effective Spread = 2 * |Trade Price - Mid Price|

        For a buy order: 2 * (Trade Price - Mid Price)
        For a sell order: 2 * (Mid Price - Trade Price)

        Args:
            trade_price: Execution price
            mid_price: Mid price at time of trade
            trade_direction: 1 for buy, -1 for sell

        Returns:
            Effective spread
        """
        return 2 * trade_direction * (trade_price - mid_price)

    @staticmethod
    def effective_spread_series(
        trade_prices: pd.Series,
        mid_prices: pd.Series,
        returns: pd.Series
    ) -> pd.Series:
        """
        Calculate effective spread for a series of trades.

        Infers trade direction from price movement.

        Args:
            trade_prices: Series of trade prices
            mid_prices: Series of mid prices
            returns: Returns to infer direction

        Returns:
            Series of effective spreads
        """
        directions = np.sign(returns)
        return 2 * directions * (trade_prices - mid_prices)

    @staticmethod
    def depth_imbalance(
        bid_sizes: np.ndarray,
        ask_sizes: np.ndarray,
        n_levels: int = 5
    ) -> float:
        """
        Calculate depth imbalance across multiple levels.

        Args:
            bid_sizes: Array of bid sizes by level
            ask_sizes: Array of ask sizes by level
            n_levels: Number of levels to consider

        Returns:
            Depth imbalance [-1, 1]
        """
        n = min(n_levels, len(bid_sizes), len(ask_sizes))

        if n == 0:
            return 0.0

        total_bid = np.sum(bid_sizes[:n])
        total_ask = np.sum(ask_sizes[:n])

        return safe_divide(total_bid - total_ask, total_bid + total_ask)

    @staticmethod
    def generate_level2_features(
        order_book_df: pd.DataFrame,
        n_levels: int = 5,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Generate all Level 2 features from order book data.

        Expected columns in order_book_df:
        - bid_price_1, bid_size_1, ... bid_price_N, bid_size_N
        - ask_price_1, ask_size_1, ... ask_price_N, ask_size_N
        OR
        - best_bid, best_ask, bid_size, ask_size

        Args:
            order_book_df: DataFrame with order book data
            n_levels: Number of levels to use
            window: Window for rolling calculations

        Returns:
            DataFrame with Level 2 features
        """
        features = pd.DataFrame(index=order_book_df.index)

        # Check if we have multi-level data
        has_multi_level = 'bid_price_1' in order_book_df.columns

        if has_multi_level:
            # Multi-level order book data
            for i, row in order_book_df.iterrows():
                bid_prices = np.array([row.get(f'bid_price_{j}', np.nan) for j in range(1, n_levels + 1)])
                bid_sizes = np.array([row.get(f'bid_size_{j}', 0) for j in range(1, n_levels + 1)])
                ask_prices = np.array([row.get(f'ask_price_{j}', np.nan) for j in range(1, n_levels + 1)])
                ask_sizes = np.array([row.get(f'ask_size_{j}', 0) for j in range(1, n_levels + 1)])

                # Remove NaN values
                valid_bids = ~np.isnan(bid_prices)
                valid_asks = ~np.isnan(ask_prices)

                if np.any(valid_bids) and np.any(valid_asks):
                    # OBI at different levels
                    for level in [1, 3, 5]:
                        if level <= n_levels:
                            features.loc[i, f'obi_{level}'] = Level2Features.order_book_imbalance_multi_level(
                                bid_prices[valid_bids], bid_sizes[valid_bids],
                                ask_prices[valid_asks], ask_sizes[valid_asks],
                                n_levels=level
                            )

                    # Weighted mid-price
                    features.loc[i, 'micro_price'] = Level2Features.weighted_mid_price_multi_level(
                        bid_prices[valid_bids], bid_sizes[valid_bids],
                        ask_prices[valid_asks], ask_sizes[valid_asks],
                        n_levels=n_levels
                    )

                    # Depth imbalance
                    features.loc[i, 'depth_imbalance'] = Level2Features.depth_imbalance(
                        bid_sizes[valid_bids], ask_sizes[valid_asks], n_levels
                    )

                    # Total depth
                    features.loc[i, 'total_bid_depth'] = np.sum(bid_sizes[valid_bids])
                    features.loc[i, 'total_ask_depth'] = np.sum(ask_sizes[valid_asks])

                    # Spread
                    features.loc[i, 'spread'] = ask_prices[0] - bid_prices[0] if len(ask_prices) > 0 and len(bid_prices) > 0 else np.nan
                    features.loc[i, 'spread_bps'] = Level2Features.bid_ask_spread(
                        bid_prices[0], ask_prices[0], in_bps=True
                    ) if len(ask_prices) > 0 and len(bid_prices) > 0 else np.nan

        else:
            # Simple best bid/ask data
            if 'best_bid' in order_book_df.columns and 'best_ask' in order_book_df.columns:
                best_bid = order_book_df['best_bid']
                best_ask = order_book_df['best_ask']
                bid_size = order_book_df.get('bid_size', pd.Series(1, index=order_book_df.index))
                ask_size = order_book_df.get('ask_size', pd.Series(1, index=order_book_df.index))

                # OBI
                features['obi'] = Level2Features.order_book_imbalance(bid_size, ask_size)

                # Micro-price
                features['micro_price'] = Level2Features.weighted_mid_price(
                    best_bid, best_ask, bid_size, ask_size
                )

                # Spread
                features['spread'] = best_ask - best_bid
                features['mid_price'] = (best_bid + best_ask) / 2
                features['spread_bps'] = (features['spread'] / features['mid_price']) * 10000

        # Rolling features
        if 'spread' in features.columns:
            features['spread_volatility'] = Level2Features.bid_ask_spread_volatility(
                features['spread'], window
            )
            features['spread_ma'] = features['spread'].rolling(window).mean()
            features['spread_zscore'] = (
                (features['spread'] - features['spread_ma']) /
                features['spread_volatility']
            )

        if 'obi' in features.columns or 'obi_1' in features.columns:
            obi_col = 'obi' if 'obi' in features.columns else 'obi_1'
            features['obi_ma'] = features[obi_col].rolling(window).mean()
            features['obi_volatility'] = features[obi_col].rolling(window).std()

        if 'depth_imbalance' in features.columns:
            features['depth_imbalance_ma'] = features['depth_imbalance'].rolling(window).mean()

        return features

    @staticmethod
    def estimate_level2_from_ohlcv(
        df: pd.DataFrame,
        spread_method: str = 'corwin_schultz'
    ) -> pd.DataFrame:
        """
        Estimate Level 2 features from OHLCV data when order book is unavailable.

        Uses various spread estimators to approximate bid/ask.

        Args:
            df: OHLCV DataFrame
            spread_method: Method for spread estimation

        Returns:
            DataFrame with estimated Level 2 features
        """
        features = pd.DataFrame(index=df.index)

        # Estimate spread using Corwin-Schultz or other methods
        if spread_method == 'corwin_schultz':
            spread = MicrostructureFeatures._corwin_schultz_spread(
                df['high'], df['low'], window=20
            )
        elif spread_method == 'parkinson':
            spread = MicrostructureFeatures._parkinson_spread(df['high'], df['low'])
        else:
            spread = MicrostructureFeatures._roll_spread(df['close'], window=20)

        features['estimated_spread'] = spread
        features['estimated_spread_bps'] = (spread / df['close']) * 10000

        # Estimate mid-price (use close as proxy)
        features['estimated_mid'] = df['close']

        # Estimate bid/ask from spread
        half_spread = spread / 2
        features['estimated_bid'] = df['close'] - half_spread
        features['estimated_ask'] = df['close'] + half_spread

        # Use volume to estimate order book imbalance
        # Positive returns with high volume = buying pressure
        returns = df['close'].pct_change()
        vol_zscore = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()

        features['estimated_obi'] = np.tanh(returns * vol_zscore)

        # Spread volatility
        features['spread_volatility'] = features['estimated_spread'].rolling(20).std()

        return features
