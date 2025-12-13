"""
Cross-Asset Feature Engineering
JPMorgan-Level Multi-Asset Analysis

Features:
- Correlation analysis
- Sector momentum
- Market breadth indicators
- Beta and factor exposures
- Pairs trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..utils.logger import get_logger
from ..utils.helpers import safe_divide


logger = get_logger(__name__)


@dataclass
class SectorData:
    """Sector classification data"""
    name: str
    symbols: List[str]
    weight_limit: float


class CrossAssetFeatures:
    """
    Cross-asset feature generator.

    Generates features based on relationships between assets:
    - Correlation dynamics
    - Relative strength
    - Sector analysis
    - Market regime
    """

    def __init__(
        self,
        sector_mapping: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize CrossAssetFeatures.

        Args:
            sector_mapping: Dictionary mapping sector names to symbols
        """
        self.sector_mapping = sector_mapping or {}
        self._correlation_cache: Dict[str, pd.DataFrame] = {}

    def calculate_rolling_correlations(
        self,
        returns: pd.DataFrame,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate rolling pairwise correlations.

        Args:
            returns: DataFrame of returns (symbols as columns)
            window: Rolling window size

        Returns:
            DataFrame with rolling correlations
        """
        n_assets = len(returns.columns)
        correlations = {}

        for i, col1 in enumerate(returns.columns):
            for col2 in returns.columns[i+1:]:
                key = f'corr_{col1}_{col2}'
                correlations[key] = returns[col1].rolling(window).corr(returns[col2])

        return pd.DataFrame(correlations, index=returns.index)

    def calculate_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """
        Calculate rolling beta against market.

        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            window: Rolling window

        Returns:
            Rolling beta series
        """
        covariance = asset_returns.rolling(window).cov(market_returns)
        variance = market_returns.rolling(window).var()

        beta = safe_divide(covariance, variance)

        return beta

    def calculate_alpha(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: float = 0.0,
        window: int = 60
    ) -> pd.Series:
        """
        Calculate rolling Jensen's alpha.

        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            risk_free_rate: Risk-free rate (annualized)
            window: Rolling window

        Returns:
            Rolling alpha series
        """
        # Convert annual rate to period rate
        rf = risk_free_rate / (252 * 26)  # For 15-min bars

        beta = self.calculate_beta(asset_returns, market_returns, window)

        # Alpha = R_asset - (Rf + Beta * (R_market - Rf))
        expected_return = rf + beta * (market_returns - rf)
        alpha = asset_returns.rolling(window).mean() - expected_return.rolling(window).mean()

        return alpha

    def relative_strength_index(
        self,
        asset_prices: pd.Series,
        benchmark_prices: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate relative strength vs benchmark.

        Args:
            asset_prices: Asset price series
            benchmark_prices: Benchmark price series
            period: RSI period

        Returns:
            Relative strength index
        """
        # Relative price ratio
        ratio = asset_prices / benchmark_prices

        # Calculate RSI of the ratio
        delta = ratio.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = safe_divide(avg_gain, avg_loss)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def sector_momentum(
        self,
        prices: pd.DataFrame,
        sector_mapping: Optional[Dict[str, List[str]]] = None,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Calculate sector momentum scores.

        Args:
            prices: DataFrame of prices (symbols as columns)
            sector_mapping: Sector to symbols mapping
            lookback: Lookback period

        Returns:
            DataFrame with sector momentum for each symbol
        """
        sector_mapping = sector_mapping or self.sector_mapping

        features = pd.DataFrame(index=prices.index)

        for sector, symbols in sector_mapping.items():
            # Get symbols that exist in prices
            valid_symbols = [s for s in symbols if s in prices.columns]

            if not valid_symbols:
                continue

            # Calculate sector return
            sector_returns = prices[valid_symbols].pct_change()
            sector_momentum = sector_returns.mean(axis=1).rolling(lookback).sum()

            features[f'sector_momentum_{sector}'] = sector_momentum

            # Rank within sector
            for symbol in valid_symbols:
                symbol_return = prices[symbol].pct_change().rolling(lookback).sum()
                features[f'{symbol}_sector_rank'] = sector_momentum.rank(pct=True)

        return features

    def market_breadth(
        self,
        prices: pd.DataFrame,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Calculate market breadth indicators.

        Args:
            prices: DataFrame of prices
            lookback: Lookback period

        Returns:
            DataFrame with breadth indicators
        """
        features = pd.DataFrame(index=prices.index)
        returns = prices.pct_change()

        # Advance/Decline ratio
        advancing = (returns > 0).sum(axis=1)
        declining = (returns < 0).sum(axis=1)
        features['adv_dec_ratio'] = safe_divide(advancing, declining)

        # Advance/Decline line
        features['adv_dec_line'] = (advancing - declining).cumsum()

        # % above moving average
        for period in [20, 50, 200]:
            above_ma = (prices > prices.rolling(period).mean()).sum(axis=1)
            features[f'pct_above_ma_{period}'] = above_ma / len(prices.columns)

        # New highs/lows
        high_52w = prices.rolling(252 * 26).max()  # ~1 year of 15-min bars
        low_52w = prices.rolling(252 * 26).min()

        new_highs = (prices >= high_52w).sum(axis=1)
        new_lows = (prices <= low_52w).sum(axis=1)

        features['new_high_low_ratio'] = safe_divide(new_highs, new_highs + new_lows)

        # McClellan Oscillator (simplified)
        adv_dec_diff = advancing - declining
        ema_19 = adv_dec_diff.ewm(span=19).mean()
        ema_39 = adv_dec_diff.ewm(span=39).mean()
        features['mcclellan_osc'] = ema_19 - ema_39

        # McClellan Summation Index
        features['mcclellan_sum'] = features['mcclellan_osc'].cumsum()

        return features

    def dispersion_index(
        self,
        returns: pd.DataFrame,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate return dispersion across assets.

        High dispersion suggests stock-picking opportunities.
        """
        # Cross-sectional standard deviation
        dispersion = returns.std(axis=1)

        return dispersion.rolling(window).mean()

    def correlation_structure(
        self,
        returns: pd.DataFrame,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate correlation structure metrics.

        Args:
            returns: DataFrame of returns
            window: Rolling window

        Returns:
            DataFrame with correlation metrics
        """
        features = pd.DataFrame(index=returns.index)

        # Average pairwise correlation - calculated properly for multi-column DataFrame
        avg_corr_values = []
        for i in range(len(returns)):
            if i < window - 1:
                avg_corr_values.append(np.nan)
            else:
                window_data = returns.iloc[i - window + 1:i + 1]
                corr_matrix = window_data.corr()
                # Get upper triangle (excluding diagonal)
                upper = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                avg_corr = upper.stack().mean()
                avg_corr_values.append(avg_corr)

        features['avg_correlation'] = avg_corr_values

        # Correlation with first principal component
        def pca_correlation(data):
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                pc1 = pca.fit_transform(data.dropna())
                explained_var = pca.explained_variance_ratio_[0]
                return explained_var
            except:
                return 0.5

        # Simplified - use variance ratio
        total_var = returns.rolling(window).var().mean(axis=1)
        portfolio_var = returns.mean(axis=1).rolling(window).var()
        features['diversification_ratio'] = safe_divide(total_var, portfolio_var)

        return features

    def pairs_trading_signals(
        self,
        prices: pd.DataFrame,
        pairs: List[Tuple[str, str]],
        lookback: int = 60,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate pairs trading signals.

        Args:
            prices: DataFrame of prices
            pairs: List of (symbol1, symbol2) tuples
            lookback: Lookback for spread calculation
            entry_zscore: Z-score threshold for entry
            exit_zscore: Z-score threshold for exit

        Returns:
            DataFrame with pairs signals
        """
        features = pd.DataFrame(index=prices.index)

        for sym1, sym2 in pairs:
            if sym1 not in prices.columns or sym2 not in prices.columns:
                continue

            # Calculate spread (log prices)
            log_p1 = np.log(prices[sym1])
            log_p2 = np.log(prices[sym2])

            # Estimate hedge ratio
            spread = log_p1.rolling(lookback).corr(log_p2) * \
                     log_p1.rolling(lookback).std() / log_p2.rolling(lookback).std()

            # Calculate spread
            spread_series = log_p1 - spread * log_p2

            # Z-score
            zscore = (spread_series - spread_series.rolling(lookback).mean()) / \
                     spread_series.rolling(lookback).std()

            pair_key = f'{sym1}_{sym2}'
            features[f'pair_spread_{pair_key}'] = spread_series
            features[f'pair_zscore_{pair_key}'] = zscore

            # Signals
            features[f'pair_signal_{pair_key}'] = np.where(
                zscore > entry_zscore, -1,  # Short spread
                np.where(zscore < -entry_zscore, 1,  # Long spread
                        np.where(abs(zscore) < exit_zscore, 0, np.nan))  # Exit
            )

            # Forward fill signals
            features[f'pair_signal_{pair_key}'] = features[f'pair_signal_{pair_key}'].ffill()

        return features

    def factor_exposures(
        self,
        returns: pd.DataFrame,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Estimate factor exposures (simplified Fama-French factors).

        Args:
            returns: DataFrame of returns
            window: Rolling window

        Returns:
            DataFrame with factor exposures
        """
        features = pd.DataFrame(index=returns.index)

        # Market factor (equal-weighted market return)
        market_return = returns.mean(axis=1)

        # Size factor (simplified - large cap vs small cap)
        # Using column order as proxy for market cap
        n_cols = len(returns.columns)
        large_cap = returns.iloc[:, :n_cols//3].mean(axis=1)
        small_cap = returns.iloc[:, -n_cols//3:].mean(axis=1)
        smb_factor = small_cap - large_cap  # Small Minus Big

        # Momentum factor
        def momentum_score(prices, lookback=252):
            return prices.pct_change(lookback)

        # Value factor (simplified using price momentum as proxy)
        # In practice, would use P/E, P/B ratios

        features['market_factor'] = market_return
        features['smb_factor'] = smb_factor

        # For each asset, calculate factor betas
        for col in returns.columns:
            features[f'{col}_market_beta'] = self.calculate_beta(
                returns[col], market_return, window
            )
            features[f'{col}_smb_beta'] = self.calculate_beta(
                returns[col], smb_factor, window
            )

        return features

    def generate_all_features(
        self,
        prices: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None,
        pairs: Optional[List[Tuple[str, str]]] = None,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Generate all cross-asset features.

        Args:
            prices: DataFrame of prices
            returns: DataFrame of returns (optional, will calculate)
            pairs: List of pairs for pairs trading
            window: Rolling window

        Returns:
            DataFrame with all cross-asset features
        """
        if returns is None:
            returns = prices.pct_change()

        features = pd.DataFrame(index=prices.index)

        # Market breadth
        breadth = self.market_breadth(prices, window // 3)
        features = pd.concat([features, breadth], axis=1)

        # Dispersion
        features['dispersion'] = self.dispersion_index(returns, window // 3)

        # Correlation structure
        corr_struct = self.correlation_structure(returns, window)
        features = pd.concat([features, corr_struct], axis=1)

        # Sector momentum if mapping available
        if self.sector_mapping:
            sector_mom = self.sector_momentum(prices, lookback=window // 3)
            features = pd.concat([features, sector_mom], axis=1)

        # Pairs trading if pairs provided
        if pairs:
            pairs_signals = self.pairs_trading_signals(prices, pairs, window)
            features = pd.concat([features, pairs_signals], axis=1)

        # Factor exposures
        factors = self.factor_exposures(returns, window)
        features = pd.concat([features, factors], axis=1)

        return features
