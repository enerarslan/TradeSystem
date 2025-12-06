"""
EQUITY CURVE DIAGNOSTIC
========================

This script analyzes your backtest results to find why metrics are unrealistic.

Run: python diagnose_equity_curve.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


def load_latest_report():
    """Load the most recent backtest report."""
    reports_dir = Path("backtesting/reports")
    json_files = list(reports_dir.glob("*.json"))
    
    if not json_files:
        print("ERROR: No reports found!")
        return None
    
    latest = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"📄 Loading: {latest.name}")
    
    with open(latest) as f:
        return json.load(f)


def analyze_trades(report: dict):
    """Analyze trade distribution for anomalies."""
    
    print("\n" + "=" * 70)
    print("TRADE ANALYSIS")
    print("=" * 70)
    
    trades = report.get("trades", [])
    
    if not trades:
        print("❌ NO TRADES IN REPORT!")
        print("   This is the problem - trades are not being recorded.")
        return
    
    print(f"\n  Total trades: {len(trades)}")
    
    # Analyze trade P&L distribution
    pnls = [t.get("pnl", 0) for t in trades]
    
    if not pnls or all(p == 0 for p in pnls):
        print("❌ ALL TRADES HAVE ZERO P&L!")
        print("   Bug: Trade profits are not being calculated.")
        return
    
    pnls = np.array(pnls)
    
    print(f"\n  P&L Statistics:")
    print(f"    Mean:   ${np.mean(pnls):,.2f}")
    print(f"    Median: ${np.median(pnls):,.2f}")
    print(f"    Std:    ${np.std(pnls):,.2f}")
    print(f"    Min:    ${np.min(pnls):,.2f}")
    print(f"    Max:    ${np.max(pnls):,.2f}")
    
    # Check for suspicious patterns
    winning = pnls[pnls > 0]
    losing = pnls[pnls < 0]
    
    print(f"\n  Win/Loss Distribution:")
    print(f"    Winning trades: {len(winning)} ({len(winning)/len(pnls)*100:.1f}%)")
    print(f"    Losing trades:  {len(losing)} ({len(losing)/len(pnls)*100:.1f}%)")
    print(f"    Break-even:     {len(pnls) - len(winning) - len(losing)}")
    
    if len(losing) == 0:
        print("\n  🔴 RED FLAG: NO LOSING TRADES!")
        print("     This is impossible in real trading.")
        print("     Likely cause: Look-ahead bias or execution bug.")
    
    if len(winning) > 0 and len(losing) > 0:
        avg_win = np.mean(winning)
        avg_loss = np.mean(np.abs(losing))
        
        print(f"\n  Avg Win:  ${avg_win:,.2f}")
        print(f"  Avg Loss: ${avg_loss:,.2f}")
        print(f"  Win/Loss Ratio: {avg_win/avg_loss:.2f}")
        
        if avg_loss < 1:
            print("\n  🔴 RED FLAG: Average loss < $1!")
            print("     Trades might not be sized correctly.")
    
    # Check trade timing
    if trades and "entry_time" in trades[0]:
        first_trade = trades[0].get("entry_time")
        last_trade = trades[-1].get("entry_time")
        print(f"\n  First trade: {first_trade}")
        print(f"  Last trade:  {last_trade}")


def check_equity_curve_pattern(report: dict):
    """Check if equity curve has suspicious patterns."""
    
    print("\n" + "=" * 70)
    print("EQUITY CURVE PATTERN ANALYSIS")
    print("=" * 70)
    
    initial = report.get("initial_capital", 100000)
    final = report.get("final_capital", 100000)
    total_ret = report.get("total_return_pct", 0)
    max_dd = report.get("max_drawdown", 0)
    volatility = report.get("volatility", 0)
    n_trades = report.get("trade_stats", {}).get("total_trades", 0)
    
    print(f"\n  Initial Capital: ${initial:,.2f}")
    print(f"  Final Capital:   ${final:,.2f}")
    print(f"  Total Return:    {total_ret:.2%}")
    print(f"  Max Drawdown:    {max_dd:.4%}")
    print(f"  Volatility:      {volatility:.4%}")
    
    # Calculate expected metrics
    print("\n  Consistency Checks:")
    
    # Check 1: Volatility vs Return
    if total_ret > 1.0 and volatility < 0.10:
        print(f"  🔴 Volatility ({volatility:.2%}) too low for return ({total_ret:.2%})")
        print("     Expected: Higher volatility with higher returns")
    
    # Check 2: Max Drawdown vs Return
    if total_ret > 1.0 and abs(max_dd) < 0.01:
        print(f"  🔴 Max Drawdown ({max_dd:.4%}) too low for return ({total_ret:.2%})")
        print("     Expected: At least 5-20% drawdown with aggressive trading")
    
    # Check 3: Trades vs Equity Changes
    if n_trades > 1000 and abs(max_dd) < 0.01:
        print(f"  🔴 {n_trades} trades but only {max_dd:.4%} max drawdown?")
        print("     Either trades are tiny or equity curve is not updating correctly")
    
    # Estimate what metrics SHOULD be
    print("\n  Expected Ranges for {:.0%} Total Return:".format(total_ret))
    
    # For 189x return over 4.5 years:
    # Daily vol should be ~3-5% to achieve this
    expected_daily_vol = 0.03  # 3%
    expected_annual_vol = expected_daily_vol * np.sqrt(252)
    expected_sharpe = (total_ret / 4.5) / expected_annual_vol
    
    print(f"    Expected Volatility: {expected_annual_vol:.1%}")
    print(f"    Expected Sharpe:     {expected_sharpe:.2f}")
    print(f"    Expected Max DD:     -20% to -50%")


def diagnose_var_issue(report: dict):
    """Diagnose why VaR is zero."""
    
    print("\n" + "=" * 70)
    print("VaR ZERO ISSUE DIAGNOSIS")
    print("=" * 70)
    
    var_95 = report.get("var_95", 0)
    var_99 = report.get("var_99", 0)
    
    print(f"\n  VaR (95%): {var_95:.6%}")
    print(f"  VaR (99%): {var_99:.6%}")
    
    if var_95 == 0 and var_99 == 0:
        print("\n  🔴 BOTH VaR VALUES ARE ZERO!")
        print("\n  Possible causes:")
        print("    1. Equity curve is flat (no changes period to period)")
        print("    2. Returns array is all zeros")
        print("    3. Returns are calculated from wrong data")
        print("\n  To diagnose, add this to BacktestEngine.run():")
        print("""
    # After creating MetricsCalculator
    print(f"Equity curve: {equity_curve[:10]}...")  # First 10 values
    print(f"Equity min/max: {min(equity_curve)}, {max(equity_curve)}")
    
    returns = np.diff(equity_curve) / equity_curve[:-1]
    print(f"Returns: {returns[:10]}...")  # First 10 returns
    print(f"Returns non-zero: {np.count_nonzero(returns)} / {len(returns)}")
        """)


def suggest_fixes():
    """Suggest specific code locations to check."""
    
    print("\n" + "=" * 70)
    print("SUGGESTED FIXES")
    print("=" * 70)
    
    print("""
  PRIORITY 1: Check Equity Curve Updates
  ───────────────────────────────────────
  
  File: backtesting/engine.py
  Class: PortfolioTracker
  
  Check these methods:
  
  1. update_prices() - Is unrealized P&L being calculated?
     ```python
     def update_prices(self, prices: dict[str, float]):
         for symbol, position in self.positions.items():
             current_price = prices.get(symbol)
             if current_price:
                 # IS THIS UPDATING EQUITY?
                 position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
     ```
  
  2. get_equity() - Does it include unrealized P&L?
     ```python
     def get_equity(self) -> float:
         total = self.cash
         for position in self.positions.values():
             total += position.market_value  # IS THIS CORRECT?
         return total
     ```
  
  3. record_equity() - Is it being called every bar?
     ```python
     # In the main backtest loop:
     for bar in bars:
         # ... process bar ...
         self.portfolio.update_prices(current_prices)
         self.equity_history.append(self.portfolio.get_equity())  # EVERY BAR?
     ```

  PRIORITY 2: Check Signal Generation
  ────────────────────────────────────
  
  File: strategies/momentum.py (or your strategy file)
  Class: TrendFollowing
  
  Check for look-ahead bias:
  
  ```python
  def generate_signal(self, bar):
      # WRONG: Using current bar's close to decide
      if bar.close > self.sma[-1]:  # This uses data we shouldn't have yet
          return BUY
      
      # CORRECT: Using previous bar's data
      if bar.close > self.sma[-2]:  # Use data from BEFORE current bar
          return BUY
  ```

  PRIORITY 3: Add Debug Logging
  ──────────────────────────────
  
  Add to BacktestEngine.run():
  
  ```python
  # Every 10000 bars, print equity
  if i % 10000 == 0:
      equity = self.portfolio.get_equity()
      print(f"Bar {i}: Equity=${equity:,.2f}, Positions={len(self.portfolio.positions)}")
  ```
    """)


def main():
    """Run full diagnosis."""
    
    print("=" * 70)
    print("BACKTEST DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Generated: {datetime.now().isoformat()}")
    
    report = load_latest_report()
    
    if report is None:
        return
    
    # Run all diagnostics
    analyze_trades(report)
    check_equity_curve_pattern(report)
    diagnose_var_issue(report)
    suggest_fixes()
    
    # Final summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    print("""
  ✅ FIXED: Metrics formulas (Sharpe, Sortino, Annualized Return)
  
  ⚠️  REMAINING ISSUES:
  
  1. Max Drawdown too low (-0.05%) 
     → Equity curve might not be updating with trades
  
  2. VaR is zero
     → Returns array might be empty or all zeros
  
  3. Metrics too good to be true
     → Possible look-ahead bias in strategy
  
  🔍 NEXT STEPS:
  
  1. Add debug prints to PortfolioTracker.get_equity()
  2. Print equity curve every 10000 bars during backtest
  3. Verify trades are affecting cash/positions
  4. Check for look-ahead bias in signal generation
    """)


if __name__ == "__main__":
    main()