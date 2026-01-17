# Trading-System Migration Report

## Migration Summary

**Date:** 2026-01-17
**Status:** ✅ COMPLETED SUCCESSFULLY

The Trading-System project has been reorganized into a professional, modular structure optimized for local execution on a 16-core workstation.

---

## 1. Files Moved

### strategies/ → models/strategies/
| Old Location | New Location |
|--------------|--------------|
| `strategies/__init__.py` | `models/strategies/__init__.py` |
| `strategies/donchian.py` | `models/strategies/donchian.py` |
| `strategies/donchian_aft_loss.py` | `models/strategies/donchian_aft_loss.py` |
| `strategies/hawkes.py` | `models/strategies/hawkes.py` |
| `strategies/moving_average.py` | `models/strategies/moving_average.py` |
| `strategies/tree_strat.py` | `models/strategies/tree_strat.py` |

### filters/ → models/filters/
| Old Location | New Location |
|--------------|--------------|
| `filters/__init__.py` | `models/filters/__init__.py` |

### visualize/ → visualization/non_interactive/
| Old Location | New Location |
|--------------|--------------|
| `visualize/__init__.py` | `visualization/non_interactive/__init__.py` |
| `visualize/run_strategy.py` | `visualization/non_interactive/run_strategy.py` |
| `visualize/hawkes_complete_analysis.py` | `visualization/non_interactive/hawkes_complete_analysis.py` |
| `visualize/hawkes_interactive_plot.py` | `visualization/non_interactive/hawkes_interactive_plot.py` |
| `visualize/output/` | `visualization/non_interactive/output/` |

### mcpt/ → backtest/mcpt/ and scripts/
| Old Location | New Location |
|--------------|--------------|
| `mcpt/insample_permutation.py` | `scripts/insample_permutation.py` |
| `mcpt/walkforward_permutation.py` | `scripts/walkforward_permutation.py` |
| `mcpt/bar_permute.py` | `backtest/mcpt/bar_permute.py` |
| `mcpt/old_strategies/` | `backtest/mcpt/old_strategies/` |
| `mcpt/other/` | `backtest/mcpt/other/` |
| `mcpt/output/` | `backtest/mcpt/output/` |
| `mcpt/README.md` | `backtest/mcpt/README.md` |
| `mcpt/LICENSE` | `backtest/mcpt/LICENSE` |
| `mcpt/.gitignore` | `backtest/mcpt/.gitignore` |

### mcpt/data/ → data/BTCUSD/
| Old Location | New Location |
|--------------|--------------|
| `mcpt/data/BTCUSD3600.pq` | `data/BTCUSD/BTCUSD3600.pq` |
| `mcpt/data/bitcoin_hourly.csv` | `data/BTCUSD/bitcoin_hourly.csv` |

### Documentation Folders → documentation/
| Old Location | New Location |
|--------------|--------------|
| `PermutationEntropy/` | `documentation/PermutationEntropy/` |
| `TradeDependenceRunsTest/` | `documentation/TradeDependenceRunTest/` |
| `VolatilityHawkes/` | `documentation/VolatilityHawkes/` |

---

## 2. Python Import Updates

### scripts/insample_permutation.py
```python
# OLD:
from config.path import (BITCOIN_CSV, BITCOIN_PARQUET, get_plot_path, ensure_directories)
from bar_permute import get_permutation
importlib.import_module(f'strategies.{strategy_name}')

# NEW:
from config.paths import (BITCOIN_CSV, BITCOIN_PARQUET, BACKTEST_FIGURES, get_plot_path, ensure_directories)
from backtest.mcpt.bar_permute import get_permutation
importlib.import_module(f'models.strategies.{strategy_name}')
```

### scripts/walkforward_permutation.py
```python
# OLD:
from config.path import (BITCOIN_PARQUET, get_plot_path, ensure_directories)
from bar_permute import get_permutation
importlib.import_module(f'strategies.{strategy_name}')

# NEW:
from config.paths import (BITCOIN_PARQUET, BACKTEST_FIGURES, get_plot_path, ensure_directories)
from backtest.mcpt.bar_permute import get_permutation
importlib.import_module(f'models.strategies.{strategy_name}')
```

### visualization/non_interactive/run_strategy.py
```python
# OLD:
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.path import BITCOIN_PARQUET, ensure_directories
output_dir = Path(__file__).resolve().parent.parent / "output" / strategy_name
importlib.import_module(f'strategies.{strategy_name}')

# NEW:
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.paths import BITCOIN_PARQUET, BACKTEST_FIGURES, ensure_directories
output_dir = BACKTEST_FIGURES / strategy_name
importlib.import_module(f'models.strategies.{strategy_name}')
```

---

## 3. Relative File Path Updates

### Output Paths (scripts/insample_permutation.py, scripts/walkforward_permutation.py)
```python
# OLD:
output_dir = Path(__file__).resolve().parent.parent / "output" / strategy_name_clean

# NEW:
output_dir = BACKTEST_FIGURES / strategy_name_clean
```

### Data Paths (config/paths.py)
```python
# OLD (config/path.py):
DATA_DIR = MCPT_DIR / "data"
BITCOIN_CSV = DATA_DIR / "bitcoin_hourly.csv"
BITCOIN_PARQUET = DATA_DIR / "BTCUSD3600.pq"

# NEW (config/paths.py):
DATA_DIR = PROJECT_ROOT / "data"
BTCUSD_DATA_DIR = DATA_DIR / "BTCUSD"
BITCOIN_CSV = BTCUSD_DATA_DIR / "bitcoin_hourly.csv"
BITCOIN_PARQUET = BTCUSD_DATA_DIR / "BTCUSD3600.pq"
```

---

## 4. New Files Created

### Configuration
- `config/paths.py` - Updated path configuration for new structure
- `config/parallel_config.yaml` - Configuration for 16-core workstation

### Orchestration Module
- `orchestration/__init__.py` - Module entry point
- `orchestration/base_orchestrator.py` - Abstract base class
- `orchestration/sequential_runner.py` - Sequential execution for debugging
- `orchestration/multiprocess_runner.py` - Multiprocessing for CPU-bound tasks
- `orchestration/dask_runner.py` - Dask for advanced workflows
- `orchestration/orchestrator_factory.py` - Factory function

### Visualization
- `visualization/__init__.py` - Module entry point
- `visualization/interactive/__init__.py` - Interactive module
- `visualization/interactive/lightweight_charts_viewer.py` - Lightweight Charts implementation
- `visualization/interactive/dashboard.py` - MCPT results dashboard
- `visualization/utils/__init__.py` - Utils module
- `visualization/utils/plotting_utils.py` - Shared plotting utilities

### Scripts
- `scripts/__init__.py` - Module entry point
- `scripts/run_parallel_backtest.py` - Main parallel execution script
- `scripts/test_migration.py` - Migration validation script

### Other
- `models/__init__.py` - Models module entry point
- `backtest/__init__.py` - Backtest module entry point
- `backtest/mcpt/__init__.py` - MCPT module entry point
- `backtest/bootstrap/.gitkeep` - Placeholder for empty directory
- `screening/.gitkeep` - Placeholder for empty directory

---

## 5. Final Directory Structure

```
Trading-system/
├── config/
│   ├── __init__.py
│   ├── path.py (legacy, for compatibility)
│   ├── paths.py (new, primary)
│   ├── parallel_config.yaml
│   ├── example_usage.py
│   └── README.md
│
├── orchestration/
│   ├── __init__.py
│   ├── base_orchestrator.py
│   ├── sequential_runner.py
│   ├── multiprocess_runner.py
│   ├── dask_runner.py
│   └── orchestrator_factory.py
│
├── models/
│   ├── __init__.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── donchian.py
│   │   ├── donchian_aft_loss.py
│   │   ├── hawkes.py
│   │   ├── moving_average.py
│   │   └── tree_strat.py
│   └── filters/
│       └── __init__.py
│
├── scripts/
│   ├── __init__.py
│   ├── insample_permutation.py
│   ├── walkforward_permutation.py
│   ├── run_parallel_backtest.py
│   └── test_migration.py
│
├── data/
│   ├── BTCUSD/
│   │   ├── BTCUSD3600.pq
│   │   └── bitcoin_hourly.csv
│   └── ETHUSD/
│
├── backtest/
│   ├── __init__.py
│   ├── mcpt/
│   │   ├── __init__.py
│   │   ├── bar_permute.py
│   │   ├── old_strategies/
│   │   ├── other/
│   │   └── output/
│   └── bootstrap/
│       └── .gitkeep
│
├── screening/
│   └── .gitkeep
│
├── visualization/
│   ├── __init__.py
│   ├── non_interactive/
│   │   ├── __init__.py
│   │   ├── run_strategy.py
│   │   ├── hawkes_complete_analysis.py
│   │   ├── hawkes_interactive_plot.py
│   │   └── output/
│   ├── interactive/
│   │   ├── __init__.py
│   │   ├── lightweight_charts_viewer.py
│   │   └── dashboard.py
│   └── utils/
│       ├── __init__.py
│       └── plotting_utils.py
│
├── documentation/
│   ├── PermutationEntropy/
│   ├── TradeDependenceRunTest/
│   └── VolatilityHawkes/
│
├── outputs/
│   └── backtest/
│       ├── results/
│       ├── reports/
│       └── figures/
│
├── README.md
└── MIGRATION_REPORT.md
```

---

## 6. Test Results

Run `python scripts/test_migration.py` to validate the migration.

**Structure Tests:** 35/35 PASSED ✅
- All directories exist
- All __init__.py files present
- All data files accessible
- All output directories writable
- Config files present

**Import Tests:** Require pandas, numpy, yaml packages installed
- Install with: `pip install pandas numpy pyyaml matplotlib seaborn tqdm`

---

## 7. Scripts Requiring Manual Testing

After installing dependencies, test:

1. **insample_permutation.py**
   ```bash
   cd /home/aherreros/Trading-system
   python scripts/insample_permutation.py donchian
   ```

2. **walkforward_permutation.py**
   ```bash
   python scripts/walkforward_permutation.py donchian
   ```

3. **run_parallel_backtest.py**
   ```bash
   python scripts/run_parallel_backtest.py --strategy donchian --test-type insample --verbose
   ```

4. **run_strategy.py**
   ```bash
   python visualization/non_interactive/run_strategy.py donchian
   ```

---

## 8. Known Issues / Notes

1. **Documentation folder data files**: CSV data files remain in documentation folders as they may contain analysis-specific data. Main data files are in `data/BTCUSD/`.

2. **Legacy config/path.py**: Kept for backward compatibility. New code should use `config/paths.py`.

3. **HPC/CESGA dependencies removed**: Project now optimized for local 16-core workstation only.

4. **Dask optional**: Dask runner requires `pip install dask[distributed]`. Multiprocess runner is the default.

---

## 9. Usage Examples

### Using the Orchestration Layer
```python
from orchestration import get_orchestrator

# Create multiprocess orchestrator (default, optimized for 16 cores)
with get_orchestrator('multiprocess', n_workers=14) as orch:
    results = orch.backtest_parallel(
        symbols=['BTCUSD', 'ETHUSD'],
        strategy_params={'name': 'donchian'},
        start='2020-01-01',
        end='2024-01-01'
    )
```

### Importing Strategies
```python
from models.strategies import donchian

signal = donchian.signal(ohlc_data, lookback=100)
result = donchian.optimize(train_data)
```

### Using Config Paths
```python
from config.paths import DATA_DIR, BACKTEST_RESULTS, ensure_directories

ensure_directories()
data_file = DATA_DIR / "BTCUSD" / "BTCUSD3600.pq"
output_file = BACKTEST_RESULTS / "my_results.csv"
```

---

## Migration Completed Successfully ✅
