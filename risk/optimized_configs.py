"""
OPTİMİZE EDİLMİŞ RİSK KONFİGÜRASYONLARI
Farklı risk profilleri için hazır ayarlar
"""

from risk.core import RiskLimitConfig


class RiskProfiles:
    """
    Hazır risk profilleri.
    
    Kullanım:
        risk_manager = EnterpriseRiskManager(config=RiskProfiles.AGGRESSIVE)
    """
    
    # CONSERVATIVE - Çok düşük risk (mevcut ayarlar gibi)
    CONSERVATIVE = RiskLimitConfig(
        max_position_size_usd=10_000,
        max_position_size_pct=5.0,
        max_portfolio_leverage=1.0,
        max_daily_loss_pct=1.0,
        max_total_drawdown_pct=5.0,
        max_daily_trades=20,
        max_open_positions=5,
        max_var_1d=2_000,
        min_cash_reserve_pct=30.0
    )
    
    # MODERATE - Dengeli risk/getiri (ÖNERİLEN)
    MODERATE = RiskLimitConfig(
        max_position_size_usd=25_000,
        max_position_size_pct=10.0,
        max_portfolio_leverage=1.0,
        max_daily_loss_pct=3.0,
        max_total_drawdown_pct=10.0,
        max_daily_trades=200,
        max_open_positions=20,
        max_var_1d=5_000,
        min_cash_reserve_pct=10.0
    )
    
    # AGGRESSIVE - Yüksek risk/getiri
    AGGRESSIVE = RiskLimitConfig(
        max_position_size_usd=50_000,
        max_position_size_pct=15.0,
        max_portfolio_leverage=1.5,
        max_daily_loss_pct=5.0,
        max_total_drawdown_pct=15.0,
        max_daily_trades=200,
        max_open_positions=25,
        max_var_1d=10_000,
        min_cash_reserve_pct=10.0
    )
    
    # PORTFOLIO - Multi-asset için optimize edilmiş
    PORTFOLIO = RiskLimitConfig(
        max_position_size_usd=100_000,  # Yüksek limit (20 pozisyon için)
        max_position_size_pct=5.0,  # Her pozisyon max %5
        max_portfolio_leverage=1.0,
        max_daily_loss_pct=4.0,
        max_total_drawdown_pct=12.0,
        max_daily_trades=500,  # Çok fazla sembol var
        max_open_positions=20,
        max_var_1d=8_000,
        min_cash_reserve_pct=5.0  # Düşük nakit, yüksek yatırım
    )
    
    # SCALPING - Çok sık işlem
    SCALPING = RiskLimitConfig(
        max_position_size_usd=5_000,
        max_position_size_pct=3.0,
        max_portfolio_leverage=1.0,
        max_daily_loss_pct=2.0,
        max_total_drawdown_pct=8.0,
        max_daily_trades=1000,  # Çok fazla işlem
        max_open_positions=10,
        max_var_1d=3_000,
        min_cash_reserve_pct=20.0
    )
    
    # SWING - Uzun vadeli pozisyonlar
    SWING = RiskLimitConfig(
        max_position_size_usd=30_000,
        max_position_size_pct=12.0,
        max_portfolio_leverage=1.0,
        max_daily_loss_pct=4.0,
        max_total_drawdown_pct=15.0,
        max_daily_trades=50,  # Az işlem
        max_open_positions=8,
        max_var_1d=7_000,
        min_cash_reserve_pct=10.0
    )


# Kullanım örnekleri
"""
# 1. Backtest için moderate profil
from risk.optimized_configs import RiskProfiles

backtester = ProfessionalBacktester(
    symbol="AAPL",
    use_risk_management=True
)

# Risk manager'ı moderate ile değiştir
backtester.risk_manager = EnterpriseRiskManager(config=RiskProfiles.MODERATE)

# 2. Portfolio backtest için portfolio profil
portfolio_backtest = MultiAssetPortfolioBacktest(
    initial_capital=100_000,
    use_risk_management=True
)
portfolio_backtest.risk_manager = EnterpriseRiskManager(config=RiskProfiles.PORTFOLIO)

# 3. Kendi custom profil
CUSTOM = RiskLimitConfig(
    max_position_size_pct=8.0,
    max_daily_trades=150,
    # ... diğer parametreler
)
"""