#!/usr/bin/env python3
"""
Test script for verifying the project reorganization was successful.

This script tests:
1. All imports resolve correctly
2. Data files are accessible from new locations
3. Configuration paths are correct
4. Output directories can be created
5. Strategy modules can be loaded

Run this script from the project root:
    python utils/test_migration.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test results storage
TESTS_PASSED = 0
TESTS_FAILED = 0
FAILURES = []


def test_result(test_name: str, passed: bool, error_msg: str = ""):
    """Record a test result."""
    global TESTS_PASSED, TESTS_FAILED, FAILURES
    if passed:
        TESTS_PASSED += 1
        print(f"  ✓ {test_name}")
    else:
        TESTS_FAILED += 1
        FAILURES.append((test_name, error_msg))
        print(f"  ✗ {test_name}")
        if error_msg:
            print(f"    Error: {error_msg}")


def test_config_imports():
    """Test config module imports."""
    print("\n1. Testing config imports...")
    try:
        from config.paths import (
            PROJECT_ROOT, CONFIG_DIR, DATA_DIR, OUTPUTS_DIR,
            BACKTEST_RESULTS, BACKTEST_FIGURES, BACKTEST_REPORTS,
            MODELS_DIR, STRATEGIES_DIR, FILTERS_DIR,
            SCRIPTS_DIR, MCPT_DIR, VISUALIZATION_DIR,
            BITCOIN_CSV, BITCOIN_PARQUET,
            ensure_directories, get_plot_path, get_output_path, get_data_path
        )
        test_result("Import config.paths", True)
    except ImportError as e:
        test_result("Import config.paths", False, str(e))
        return

    try:
        from config import (
            PROJECT_ROOT, DATA_DIR, BACKTEST_RESULTS,
            ensure_directories, get_plot_path
        )
        test_result("Import from config (alias)", True)
    except ImportError as e:
        test_result("Import from config (alias)", False, str(e))


def test_models_imports():
    """Test models module imports."""
    print("\n2. Testing models imports...")

    # First check if pandas is available
    try:
        import pandas
        has_pandas = True
    except ImportError:
        has_pandas = False
        print("  (Skipping - pandas not installed)")

    if has_pandas:
        try:
            from models import strategies
            test_result("Import models.strategies", True)
        except ImportError as e:
            test_result("Import models.strategies", False, str(e))

        try:
            from models.strategies import donchian, moving_average, hawkes
            test_result("Import strategy modules", True)
        except ImportError as e:
            test_result("Import strategy modules", False, str(e))

        try:
            from models import indicators
            test_result("Import models.filters", True)
        except ImportError as e:
            test_result("Import models.filters", False, str(e))

        # Test strategy functions exist
        try:
            from models.strategies import donchian
            assert hasattr(donchian, 'signal'), "donchian.signal missing"
            assert hasattr(donchian, 'optimize'), "donchian.optimize missing"
            test_result("Strategy functions (signal, optimize)", True)
        except Exception as e:
            test_result("Strategy functions (signal, optimize)", False, str(e))
    else:
        # Check file structure instead
        from config.paths import STRATEGIES_DIR
        test_result("strategies/__init__.py exists",
                    (STRATEGIES_DIR / "__init__.py").exists())
        test_result("strategies/donchian.py exists",
                    (STRATEGIES_DIR / "donchian.py").exists())
        test_result("filters/__init__.py exists",
                    (PROJECT_ROOT / "models" / "filters" / "__init__.py").exists())


def test_orchestration_imports():
    """Test orchestration module imports."""
    print("\n3. Testing orchestration imports...")
    try:
        from orchestration import get_orchestrator
        test_result("Import get_orchestrator", True)
    except ImportError as e:
        test_result("Import get_orchestrator", False, str(e))

    try:
        from orchestration.base_orchestrator import BaseOrchestrator
        from orchestration.sequential_runner import SequentialRunner
        from orchestration.multiprocess_runner import MultiprocessRunner
        from orchestration.dask_runner import DaskRunner
        test_result("Import orchestrator classes", True)
    except ImportError as e:
        test_result("Import orchestrator classes", False, str(e))

    # Test factory function
    try:
        from orchestration import get_orchestrator
        orch = get_orchestrator('sequential')
        assert orch is not None
        test_result("Create sequential orchestrator", True)
    except Exception as e:
        test_result("Create sequential orchestrator", False, str(e))


def test_backtest_imports():
    """Test backtest module imports."""
    print("\n4. Testing backtest imports...")
    try:
        from backtest.mcpt.bar_permute import get_permutation
        test_result("Import bar_permute.get_permutation", True)
    except ImportError as e:
        test_result("Import bar_permute.get_permutation", False, str(e))


def test_visualization_imports():
    """Test visualization module imports."""
    print("\n5. Testing visualization imports...")
    try:
        from visualization import non_interactive, interactive, utils
        test_result("Import visualization modules", True)
    except ImportError as e:
        test_result("Import visualization modules", False, str(e))

    try:
        from visualization.interactive.dashboard import MCPTDashboard
        test_result("Import MCPTDashboard", True)
    except ImportError as e:
        test_result("Import MCPTDashboard", False, str(e))

    try:
        from visualization.utils.plotting_utils import set_plot_style, save_figure
        test_result("Import plotting_utils", True)
    except ImportError as e:
        test_result("Import plotting_utils", False, str(e))


def test_data_paths():
    """Test data file accessibility."""
    print("\n6. Testing data paths...")
    from config.paths import DATA_DIR, BITCOIN_CSV, BITCOIN_PARQUET

    test_result("DATA_DIR exists", DATA_DIR.exists(), f"Not found: {DATA_DIR}")
    test_result("BTCUSD data dir exists", (DATA_DIR / "BTCUSD").exists(),
                f"Not found: {DATA_DIR / 'BTCUSD'}")
    test_result("BITCOIN_PARQUET exists", BITCOIN_PARQUET.exists(),
                f"Not found: {BITCOIN_PARQUET}")
    test_result("BITCOIN_CSV exists", BITCOIN_CSV.exists(),
                f"Not found: {BITCOIN_CSV}")


def test_output_directories():
    """Test output directory creation."""
    print("\n7. Testing output directories...")
    from config.paths import (
        ensure_directories,
        OUTPUTS_DIR, BACKTEST_RESULTS, BACKTEST_FIGURES, BACKTEST_REPORTS
    )

    try:
        ensure_directories()
        test_result("ensure_directories() runs", True)
    except Exception as e:
        test_result("ensure_directories() runs", False, str(e))

    test_result("OUTPUTS_DIR exists", OUTPUTS_DIR.exists(), f"Not found: {OUTPUTS_DIR}")
    test_result("BACKTEST_RESULTS exists", BACKTEST_RESULTS.exists(),
                f"Not found: {BACKTEST_RESULTS}")
    test_result("BACKTEST_FIGURES exists", BACKTEST_FIGURES.exists(),
                f"Not found: {BACKTEST_FIGURES}")
    test_result("BACKTEST_REPORTS exists", BACKTEST_REPORTS.exists(),
                f"Not found: {BACKTEST_REPORTS}")


def test_write_output():
    """Test writing to output directories."""
    print("\n8. Testing write access...")
    from config.paths import BACKTEST_RESULTS

    test_file = BACKTEST_RESULTS / "test_migration_check.txt"
    try:
        with open(test_file, 'w') as f:
            f.write("Migration test successful\n")
        test_result("Write to BACKTEST_RESULTS", True)

        # Clean up
        test_file.unlink()
        test_result("Clean up test file", True)
    except Exception as e:
        test_result("Write to BACKTEST_RESULTS", False, str(e))


def test_config_parallel():
    """Test parallel config file exists."""
    print("\n9. Testing config files...")
    from config.paths import CONFIG_DIR

    yaml_file = CONFIG_DIR / "parallel_config.yaml"
    test_result("parallel_config.yaml exists", yaml_file.exists(),
                f"Not found: {yaml_file}")

    if yaml_file.exists():
        try:
            import yaml
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
            test_result("parallel_config.yaml is valid YAML", True)
        except Exception as e:
            test_result("parallel_config.yaml is valid YAML", False, str(e))


def test_directory_structure():
    """Test overall directory structure."""
    print("\n10. Testing directory structure...")
    from config.paths import PROJECT_ROOT

    required_dirs = [
        'config',
        'orchestration',
        'models/strategies',
        'models/filters',
        'scripts',
        'data/BTCUSD',
        'backtest/mcpt',
        'backtest/bootstrap',
        'screening',
        'visualization/non_interactive',
        'visualization/interactive',
        'visualization/utils',
        'documentation/PermutationEntropy',
        'documentation/TradeDependenceRunTest',
        'documentation/VolatilityHawkes',
        'outputs/backtest/results',
        'outputs/backtest/reports',
        'outputs/backtest/figures',
    ]

    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        test_result(f"Directory: {dir_path}", full_path.exists(),
                    f"Not found: {full_path}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("TRADING-SYSTEM MIGRATION TEST")
    print("=" * 70)
    print(f"\nProject root: {PROJECT_ROOT}")

    test_config_imports()
    test_models_imports()
    test_orchestration_imports()
    test_backtest_imports()
    test_visualization_imports()
    test_data_paths()
    test_output_directories()
    test_write_output()
    test_config_parallel()
    test_directory_structure()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Passed: {TESTS_PASSED}")
    print(f"  Failed: {TESTS_FAILED}")
    print(f"  Total:  {TESTS_PASSED + TESTS_FAILED}")

    if FAILURES:
        print("\n  Failed tests:")
        for test_name, error in FAILURES:
            print(f"    - {test_name}")
            if error:
                print(f"      {error}")

    print("=" * 70)

    if TESTS_FAILED == 0:
        print("\n✓ ALL TESTS PASSED - Migration successful!")
        return 0
    else:
        print(f"\n✗ {TESTS_FAILED} TESTS FAILED - Please review and fix issues")
        return 1


if __name__ == '__main__':
    sys.exit(main())
