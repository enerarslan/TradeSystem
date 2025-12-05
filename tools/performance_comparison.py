# backtest.py dosyasının EN ALTINA ekle (async def main() fonksiyonunu değiştir):

async def main():
    """
    UPDATED MAIN - Optimize edilmiş risk ile test
    """
    from risk.optimized_configs import RiskProfiles
    
    log.info("="*70)
    log.info("   🎯 OPTİMİZE EDİLMİŞ BACKTEST")
    log.info("="*70 + "\n")
    
    # Test 1: AAPL - MODERATE RISK (ÖNERİLEN)
    log.info("📊 Test 1: AAPL - MODERATE Risk Profili")
    backtester = ProfessionalBacktester(
        symbol="AAPL",
        initial_capital=100_000,
        commission_pct=0.001,
        slippage_pct=0.0005,
        use_risk_management=True
    )
    
    # Risk profilini değiştir
    backtester.risk_manager.config = RiskProfiles.MODERATE
    
    metrics = await backtester.run(
        strategy_class=AdvancedMomentum,
        strategy_params={
            'fast_period': 10,
            'slow_period': 30,
            'min_confidence': 0.5  # Daha düşük (daha fazla işlem)
        }
    )
    
    if metrics:
        log.success(f"✅ CAGR: {metrics.cagr:.2f}%, Sharpe: {metrics.sharpe_ratio:.3f}")


if __name__ == "__main__":
    asyncio.run(main())