#!/usr/bin/env python3
"""
============================================================================
ALPHATRADE - SYSTEM HEALTH CHECK
============================================================================
Run this before starting to validate your system setup.

Usage:
    python check_system.py
============================================================================
"""

import sys
import importlib
from pathlib import Path
from datetime import datetime

# Colors for terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def check_mark(passed: bool) -> str:
    if passed:
        return f"{Colors.GREEN}✅{Colors.END}"
    return f"{Colors.RED}❌{Colors.END}"


def warn_mark() -> str:
    return f"{Colors.YELLOW}⚠️{Colors.END}"


def print_header(title: str):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"   {title}")
    print(f"{'='*60}{Colors.END}\n")


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    passed = version.major >= 3 and version.minor >= 10
    print(f"  {check_mark(passed)} Python Version: {version.major}.{version.minor}.{version.micro}")
    if not passed:
        print(f"     {Colors.RED}Requires Python 3.10+{Colors.END}")
    return passed


def check_dependencies():
    """Check required packages"""
    print_header("DEPENDENCY CHECK")
    
    required = {
        'pandas': '2.0.0',
        'numpy': '1.24.0',
        'pydantic': '2.0.0',
        'loguru': '0.7.0',
        'pyyaml': '6.0.0',
    }
    
    optional = {
        'xgboost': '2.0.0',
        'scikit-learn': '1.3.0',
        'matplotlib': '3.7.0',
        'ccxt': '4.0.0',
        'websockets': '11.0.0',
        'sqlalchemy': '2.0.0',
        'aiosqlite': '0.19.0',
    }
    
    all_passed = True
    
    print("  Required Packages:")
    for pkg, min_version in required.items():
        try:
            mod = importlib.import_module(pkg.replace('-', '_'))
            version = getattr(mod, '__version__', 'unknown')
            print(f"    {check_mark(True)} {pkg}: {version}")
        except ImportError:
            print(f"    {check_mark(False)} {pkg}: NOT INSTALLED")
            all_passed = False
    
    print("\n  Optional Packages:")
    for pkg, min_version in optional.items():
        try:
            mod = importlib.import_module(pkg.replace('-', '_'))
            version = getattr(mod, '__version__', 'unknown')
            print(f"    {check_mark(True)} {pkg}: {version}")
        except ImportError:
            print(f"    {warn_mark()} {pkg}: not installed (optional)")
    
    return all_passed


def check_project_structure():
    """Check project structure"""
    print_header("PROJECT STRUCTURE CHECK")
    
    required_dirs = [
        'config',
        'core',
        'data',
        'data/storage',
        'strategies',
        'risk',
        'execution',
        'utils',
        'backtest',
        'ml',
        'logs',
        'data/backtest_results',
    ]
    
    required_files = [
        'config/settings.py',
        'config/settings.yaml',
        'config/yaml_config.py',
        'core/__init__.py',
        'core/events.py',
        'core/bus.py',
        'data/models.py',
        'data/csv_loader.py',
        'strategies/base.py',
        'strategies/momentum.py',
        'risk/core.py',
        'execution/portfolio.py',
        'utils/logger.py',
        'requirements.txt',
        'run_backtest.py',
    ]
    
    all_passed = True
    
    print("  Directories:")
    for dir_path in required_dirs:
        path = Path(dir_path)
        exists = path.exists() and path.is_dir()
        print(f"    {check_mark(exists)} {dir_path}/")
        if not exists:
            all_passed = False
    
    print("\n  Core Files:")
    for file_path in required_files:
        path = Path(file_path)
        exists = path.exists() and path.is_file()
        print(f"    {check_mark(exists)} {file_path}")
        if not exists:
            all_passed = False
    
    return all_passed


def check_data_files():
    """Check data files"""
    print_header("DATA FILES CHECK")
    
    storage_path = Path("data/storage")
    
    if not storage_path.exists():
        print(f"  {check_mark(False)} data/storage directory does not exist")
        return False
    
    csv_files = list(storage_path.glob("*_15min.csv"))
    other_csv = list(storage_path.glob("*.csv"))
    xlsx_files = list(storage_path.glob("*.xlsx"))
    
    total_files = len(csv_files) + len(other_csv) + len(xlsx_files)
    
    print(f"  {check_mark(total_files > 0)} Found {total_files} data files:")
    print(f"      - 15min CSVs: {len(csv_files)}")
    print(f"      - Other CSVs: {len(other_csv) - len(csv_files)}")
    print(f"      - Excel files: {len(xlsx_files)}")
    
    if csv_files:
        print(f"\n  Sample files (first 5):")
        for f in csv_files[:5]:
            size_kb = f.stat().st_size / 1024
            symbol = f.stem.replace("_15min", "")
            print(f"      - {symbol}: {size_kb:.1f} KB")
        if len(csv_files) > 5:
            print(f"      ... and {len(csv_files) - 5} more")
    
    return total_files > 0


def check_configuration():
    """Check configuration"""
    print_header("CONFIGURATION CHECK")
    
    try:
        from config.yaml_config import get_config
        config = get_config()
        
        print(f"  {check_mark(True)} Configuration loaded successfully")
        print(f"      - App Name: {config.app_name}")
        print(f"      - Version: {config.version}")
        print(f"      - Environment: {config.environment}")
        print(f"      - Initial Capital: ${config.trading.initial_capital:,.0f}")
        print(f"      - Max Positions: {config.risk.max_open_positions}")
        
        return True
    except Exception as e:
        print(f"  {check_mark(False)} Configuration error: {e}")
        return False


def check_imports():
    """Check if core modules can be imported"""
    print_header("MODULE IMPORT CHECK")
    
    modules_to_check = [
        ('config.settings', 'Settings'),
        ('core.events', 'Event'),
        ('core.bus', 'EventBus'),
        ('data.models', 'MarketTick'),
        ('data.csv_loader', 'LocalCSVLoader'),
        ('strategies.base', 'BaseStrategy'),
        ('strategies.momentum', 'AdvancedMomentum'),
        ('risk.core', 'EnterpriseRiskManager'),
        ('execution.portfolio', 'PortfolioManager'),
        ('utils.logger', 'log'),
    ]
    
    all_passed = True
    
    for module_path, class_name in modules_to_check:
        try:
            module = importlib.import_module(module_path)
            obj = getattr(module, class_name, None)
            if obj:
                print(f"  {check_mark(True)} {module_path}.{class_name}")
            else:
                print(f"  {check_mark(False)} {module_path}.{class_name} not found")
                all_passed = False
        except Exception as e:
            print(f"  {check_mark(False)} {module_path}: {str(e)[:50]}")
            all_passed = False
    
    return all_passed


def run_quick_test():
    """Run a quick functionality test"""
    print_header("QUICK FUNCTIONALITY TEST")
    
    try:
        # Test data loading
        from data.csv_loader import LocalCSVLoader
        loader = LocalCSVLoader()
        
        # Find first available symbol
        storage_path = Path("data/storage")
        csv_files = list(storage_path.glob("*_15min.csv"))
        
        if not csv_files:
            print(f"  {warn_mark()} No data files to test")
            return True
        
        symbol = csv_files[0].stem.replace("_15min", "")
        print(f"  Testing with symbol: {symbol}")
        
        ticks = loader.load_data(symbol, use_cache=False)
        
        if ticks:
            print(f"  {check_mark(True)} Data loading: {len(ticks)} bars loaded")
            print(f"      - First: {ticks[0].timestamp}")
            print(f"      - Last: {ticks[-1].timestamp}")
        else:
            print(f"  {check_mark(False)} Data loading failed")
            return False
        
        # Test strategy
        from strategies.momentum import AdvancedMomentum
        strategy = AdvancedMomentum(symbol=symbol)
        print(f"  {check_mark(True)} Strategy creation: {strategy.name}")
        
        # Test risk manager
        from risk.core import EnterpriseRiskManager
        risk_mgr = EnterpriseRiskManager()
        print(f"  {check_mark(True)} Risk manager initialized")
        
        return True
        
    except Exception as e:
        print(f"  {check_mark(False)} Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks"""
    print(f"\n{'='*60}")
    print(f"   🏦 ALPHATRADE SYSTEM HEALTH CHECK")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    results = {}
    
    # Run all checks
    print_header("PYTHON VERSION")
    results['python'] = check_python_version()
    
    results['dependencies'] = check_dependencies()
    results['structure'] = check_project_structure()
    results['data'] = check_data_files()
    results['config'] = check_configuration()
    results['imports'] = check_imports()
    results['test'] = run_quick_test()
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for check, result in results.items():
        print(f"  {check_mark(result)} {check.capitalize()}")
    
    print(f"\n  {Colors.BLUE}Result: {passed}/{total} checks passed{Colors.END}")
    
    if passed == total:
        print(f"\n  {Colors.GREEN}🎉 System is ready!{Colors.END}")
        print(f"  Run: python run_backtest.py --help")
    else:
        print(f"\n  {Colors.YELLOW}⚠️ Some checks failed. Please fix before running.{Colors.END}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)