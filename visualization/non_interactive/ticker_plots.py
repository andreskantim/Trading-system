#!/usr/bin/env python3
"""
Ticker-level statistics and plotting functions.

Functions:
- calculate_ticker_statistics: Returns all 12 blocks of statistics
- plot_ticker_results: Generates cumulative log with percentiles + p-value histogram
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# BLOCK 1 — PERFORMANCE
# ============================================================================

def _calc_performance(returns: np.ndarray, trades: List[float], periods_per_year: int = 8760) -> Dict:
    cum_log = float(np.nansum(returns))
    total_return_pct = (np.exp(cum_log) - 1) * 100
    n = len(returns)
    years = n / periods_per_year
    cagr = ((np.exp(cum_log) ** (1/years)) - 1) * 100 if years > 0 else 0.0
    mean_return_period = float(np.nanmean(returns))
    mean_return_trade = float(np.mean(trades)) if trades else 0.0

    return {
        'total_return_pct': total_return_pct,
        'cumulative_log_return': cum_log,
        'cagr_pct': cagr,
        'total_pnl': cum_log,
        'mean_return_period': mean_return_period,
        'mean_return_trade': mean_return_trade,
        # 'alpha': None  # Todo: requires benchmark
    }


# ============================================================================
# BLOCK 2 — RISK
# ============================================================================

def _calc_risk(returns: np.ndarray) -> Dict:
    cum = np.nancumsum(returns)
    equity = np.exp(cum)
    peak = np.maximum.accumulate(equity)
    dd_pct = (equity - peak) / peak * 100
    dd_abs = equity - peak

    max_dd_pct = float(np.nanmin(dd_pct))
    max_dd_abs = float(np.nanmin(dd_abs))

    # Max DD duration
    in_dd = equity < peak
    max_dur, cur_dur = 0, 0
    for i in range(len(in_dd)):
        cur_dur = cur_dur + 1 if in_dd[i] else 0
        max_dur = max(max_dur, cur_dur)

    ulcer = float(np.sqrt(np.mean(dd_pct ** 2)))
    mean_dd = float(np.nanmean(dd_pct[dd_pct < 0])) if np.any(dd_pct < 0) else 0.0

    return {
        'max_dd_pct': max_dd_pct,
        'max_dd_abs': max_dd_abs,
        'max_dd_duration': max_dur,
        'ulcer_index': ulcer,
        'mean_dd': mean_dd,
    }


# ============================================================================
# BLOCK 3 — VOLATILITY
# ============================================================================

def _calc_volatility(returns: np.ndarray, periods_per_year: int = 8760) -> Dict:
    std = float(np.nanstd(returns))
    neg = returns[returns < 0]
    downside_std = float(np.nanstd(neg)) if len(neg) > 1 else 0.0

    return {
        'annualized_vol_pct': std * np.sqrt(periods_per_year) * 100,
        'monthly_vol_pct': std * np.sqrt(30 * 24) * 100,
        'downside_vol_pct': downside_std * np.sqrt(periods_per_year) * 100,
        'semi_deviation': downside_std,
    }


# ============================================================================
# BLOCK 4 — RISK-RETURN RATIOS
# ============================================================================

def _calc_risk_return(returns: np.ndarray, periods_per_year: int = 8760) -> Dict:
    mean_r = np.nanmean(returns)
    std_r = np.nanstd(returns)
    neg = returns[returns < 0]
    downside_std = np.nanstd(neg) if len(neg) > 1 else 1e-10

    ann_ret = mean_r * periods_per_year
    ann_std = std_r * np.sqrt(periods_per_year)
    ann_down = downside_std * np.sqrt(periods_per_year)

    sharpe = ann_ret / ann_std if ann_std > 0 else 0.0
    sortino = ann_ret / ann_down if ann_down > 0 else 0.0

    cum = np.nancumsum(returns)
    years = len(returns) / periods_per_year
    cagr = ((np.exp(cum[-1]) ** (1/years)) - 1) * 100 if years > 0 and len(cum) > 0 else 0.0
    max_dd = abs(float(np.nanmin((np.exp(cum) - np.maximum.accumulate(np.exp(cum))) / np.maximum.accumulate(np.exp(cum)) * 100)))
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    sqn = np.sqrt(len(returns)) * mean_r / std_r if std_r > 0 and len(returns) >= 30 else 0.0

    return {
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'calmar_ratio': float(calmar),
        'sqn': float(sqn),
    }


# ============================================================================
# BLOCK 5 — TRADE STATS
# ============================================================================

def _calc_trade_stats(trades: List[float]) -> Dict:
    if not trades:
        return {'n_trades': 0, 'win_rate_pct': 0, 'avg_gain': 0, 'avg_loss': 0, 'profit_factor': 0, 'expectancy': 0}

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    return {
        'n_trades': len(trades),
        'win_rate_pct': len(wins) / len(trades) * 100,
        'avg_gain': float(np.mean(wins)) if wins else 0.0,
        'avg_loss': float(np.mean(losses)) if losses else 0.0,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0),
        'expectancy': float(np.mean(trades)),
    }


# ============================================================================
# BLOCK 6 — PNL DISTRIBUTION
# ============================================================================

def _calc_distribution(trades: List[float]) -> Dict:
    if not trades:
        return {'skewness': 0, 'kurtosis': 0, 'p5': 0, 'p25': 0, 'p50': 0, 'p75': 0, 'p95': 0, 'best_trade': 0, 'worst_trade': 0}

    arr = np.array(trades)
    return {
        'skewness': float(scipy_stats.skew(arr)),
        'kurtosis': float(scipy_stats.kurtosis(arr)),
        'p5': float(np.percentile(arr, 5)),
        'p25': float(np.percentile(arr, 25)),
        'p50': float(np.percentile(arr, 50)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
        'best_trade': float(np.max(arr)),
        'worst_trade': float(np.min(arr)),
    }


# ============================================================================
# BLOCK 7 — STATISTICAL ROBUSTNESS
# ============================================================================

def _calc_robustness(returns: np.ndarray, perm_pfs: List[float], real_pf: float) -> Dict:
    perm_better = sum(1 for pf in perm_pfs if pf >= real_pf) + 1
    p_value = perm_better / len(perm_pfs) if perm_pfs else 1.0
    percentile = (1 - p_value) * 100

    # PBO placeholder - would require proper implementation
    pbo = None

    # Viable strategies ratio
    viable_ratio = sum(1 for pf in perm_pfs if pf > 1) / len(perm_pfs) if perm_pfs else 0.0

    return {
        'p_value': p_value,
        'percentile_vs_perms': percentile,
        'pbo': pbo,
        'viable_strategies_ratio': viable_ratio,
    }


# ============================================================================
# BLOCK 8 — TEMPORAL STABILITY (BY YEAR)
# ============================================================================

def _calc_temporal_stability(returns: np.ndarray, index: pd.DatetimeIndex, signal: np.ndarray,
                             perm_pfs: List[float], real_pf: float, periods_per_year: int = 8760) -> Dict:
    df = pd.DataFrame({'returns': returns, 'signal': signal}, index=index)
    df['year'] = df.index.year

    yearly_stats = {}
    for year, grp in df.groupby('year'):
        yr_rets = grp['returns'].values
        yr_sig = grp['signal'].values

        cum = np.nancumsum(yr_rets)
        eq = np.exp(cum)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100

        std = np.nanstd(yr_rets)
        mean = np.nanmean(yr_rets)
        sharpe = (mean * periods_per_year) / (std * np.sqrt(periods_per_year)) if std > 0 else 0.0
        sqn = np.sqrt(len(yr_rets)) * mean / std if std > 0 else 0.0

        yearly_stats[int(year)] = {
            'cumulative_log': float(cum[-1]) if len(cum) > 0 else 0.0,
            'pnl': float(np.nansum(yr_rets)),
            'cagr_pct': float((np.exp(cum[-1]) - 1) * 100) if len(cum) > 0 else 0.0,
            'max_dd_pct': float(np.nanmin(dd)),
            'max_dd_duration': 0,  # Simplified
            'sharpe': float(sharpe),
            'sqn': float(sqn),
        }

    # Calculate variability across years
    if yearly_stats:
        metrics = ['cumulative_log', 'pnl', 'cagr_pct', 'max_dd_pct', 'sharpe', 'sqn']
        variability = {}
        for m in metrics:
            vals = [v[m] for v in yearly_stats.values()]
            variability[f'{m}_std'] = float(np.std(vals)) if len(vals) > 1 else 0.0
    else:
        variability = {}

    return {
        'yearly_stats': yearly_stats,
        'variability': variability,
    }


# ============================================================================
# BLOCK 9 — DEPENDENCIES
# ============================================================================

def _calc_dependencies(returns: np.ndarray) -> Dict:
    clean = returns[~np.isnan(returns)]

    def autocorr(x, lag):
        if len(x) < lag + 2:
            return 0.0
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

    # Hurst exponent (simplified R/S)
    def hurst(x, max_lag=100):
        if len(x) < max_lag * 2:
            return 0.5
        lags = range(2, min(max_lag, len(x) // 10))
        rs_vals = []
        for lag in lags:
            n_sub = len(x) // lag
            rs_sub = []
            for i in range(n_sub):
                sub = x[i*lag:(i+1)*lag]
                adj = sub - np.mean(sub)
                cum = np.cumsum(adj)
                r = np.max(cum) - np.min(cum)
                s = np.std(sub)
                if s > 0:
                    rs_sub.append(r / s)
            if rs_sub:
                rs_vals.append((lag, np.mean(rs_sub)))
        if len(rs_vals) < 3:
            return 0.5
        lags_log = np.log([x[0] for x in rs_vals])
        rs_log = np.log([x[1] for x in rs_vals])
        slope, _, _, _, _ = scipy_stats.linregress(lags_log, rs_log)
        return float(slope)

    cum = np.nancumsum(returns)
    eq = np.exp(cum)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak

    return {
        'return_autocorr_1': autocorr(clean, 1),
        'return_autocorr_5': autocorr(clean, 5),
        'dd_autocorr_1': autocorr(dd, 1) if len(dd) > 2 else 0.0,
        'hurst_exponent': hurst(clean),
        'tail_dependence': None,  # Todo
    }


# ============================================================================
# BLOCK 10 — COSTS [COMMENTED]
# ============================================================================

# def _calc_costs(signal, prices, commission_pct=0.1, slippage_pct=0.05):
#     """Todo: commissions, slippage, turnover, avg_time_in_position, exposure"""
#     pass


# ============================================================================
# BLOCK 11 — CONCENTRATION
# ============================================================================

def _calc_concentration(trades: List[float]) -> Dict:
    if not trades:
        return {'gini': 0, 'top_5pct_pnl_share': 0, 'risk_per_asset': None, 'risk_per_period': None}

    # Gini coefficient
    arr = np.abs(trades)
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    cum = np.cumsum(sorted_arr)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n if cum[-1] > 0 else 0.0

    # Top 5% trades PnL share
    top_n = max(1, int(len(trades) * 0.05))
    sorted_by_abs = sorted(trades, key=abs, reverse=True)
    top_pnl = sum(sorted_by_abs[:top_n])
    total_pnl = sum(trades)
    top_5pct_share = abs(top_pnl / total_pnl) * 100 if total_pnl != 0 else 0.0

    return {
        'gini': float(max(0, gini)),
        'top_5pct_pnl_share': float(top_5pct_share),
        'risk_per_asset': None,  # Todo: multi-asset
        'risk_per_period': None,  # Todo
    }


# ============================================================================
# BLOCK 12 — OVERFITTING [COMMENTED]
# ============================================================================

# def _calc_overfitting(train_sharpe, test_sharpe, n_params, n_combos):
#     """Todo: param_sensitivity, robustness, cross_asset_consistency, degradation"""
#     pass


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def identify_trades(signal: np.ndarray, returns: np.ndarray) -> List[float]:
    """Identify individual trades from signal and returns."""
    trades = []
    trade_pnl = 0.0
    in_trade = False
    prev_sig = 0

    for i in range(len(signal)):
        sig = signal[i]
        ret = returns[i] if not np.isnan(returns[i]) else 0.0

        if sig != 0 and not in_trade:
            in_trade = True
            trade_pnl = ret
        elif sig != 0 and in_trade:
            if sig == prev_sig:
                trade_pnl += ret
            else:
                trades.append(trade_pnl)
                trade_pnl = ret
        elif sig == 0 and in_trade:
            trades.append(trade_pnl)
            trade_pnl = 0.0
            in_trade = False

        prev_sig = sig

    if in_trade:
        trades.append(trade_pnl)

    return trades


def calculate_ticker_statistics(returns: np.ndarray, signal: np.ndarray, index: pd.DatetimeIndex,
                                perm_pfs: List[float], real_pf: float,
                                periods_per_year: int = 8760) -> Dict[str, Any]:
    """
    Calculate all 12 blocks of statistics for a single ticker.

    Args:
        returns: Strategy returns array
        signal: Trading signal array
        index: DatetimeIndex for temporal analysis
        perm_pfs: List of permutation profit factors
        real_pf: Real strategy profit factor
        periods_per_year: Periods per year (8760 for hourly)

    Returns:
        Dict with all 12 blocks of statistics
    """
    trades = identify_trades(signal, returns)

    return {
        'performance': _calc_performance(returns, trades, periods_per_year),
        'risk': _calc_risk(returns),
        'volatility': _calc_volatility(returns, periods_per_year),
        'risk_return': _calc_risk_return(returns, periods_per_year),
        'trade_stats': _calc_trade_stats(trades),
        'distribution': _calc_distribution(trades),
        'robustness': _calc_robustness(returns, perm_pfs, real_pf),
        'temporal': _calc_temporal_stability(returns, index, signal, perm_pfs, real_pf, periods_per_year),
        'dependencies': _calc_dependencies(returns),
        # 'costs': _calc_costs(...),  # BLOCK 10 - TODO
        'concentration': _calc_concentration(trades),
        # 'overfitting': _calc_overfitting(...),  # BLOCK 12 - TODO
    }


def plot_ticker_results(ticker: str, strategy: str, index: pd.DatetimeIndex,
                        real_cum_rets: np.ndarray, perm_cum_rets: List[np.ndarray],
                        perm_pfs: List[float], real_pf: float, p_value: float,
                        output_dir: Path, prefix: str = '', vlines: List[tuple] = None):
    """
    Generate plots for ticker results.

    Plots:
    1. Cumulative log returns with percentiles
    2. P-value histogram

    Args:
        vlines: Optional list of (date, color, label) tuples for vertical lines
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('dark_background')
    file_prefix = f'{ticker}_{prefix}_' if prefix else f'{ticker}_'

    # Plot 1: Cumulative Returns with Percentiles
    fig, ax = plt.subplots(figsize=(14, 8))

    perm_matrix = np.array(perm_cum_rets).T
    p5 = np.percentile(perm_matrix, 5, axis=1)
    p25 = np.percentile(perm_matrix, 25, axis=1)
    p50 = np.percentile(perm_matrix, 50, axis=1)
    p75 = np.percentile(perm_matrix, 75, axis=1)
    p95 = np.percentile(perm_matrix, 95, axis=1)

    ax.fill_between(index, p5, p95, color='white', alpha=0.1, label='5-95 percentile')
    ax.fill_between(index, p25, p75, color='white', alpha=0.2, label='25-75 percentile')
    ax.plot(index, p50, color='yellow', linewidth=1.5, alpha=0.6, label='Median perms')
    ax.plot(index, real_cum_rets, color='red', linewidth=2.5, label=f'Real (PF={real_pf:.4f})', zorder=100)

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Log Return")
    ax.set_title(f"{ticker} {prefix.title()} Cumulative Returns ({strategy}) | p={p_value:.4f}")

    # Add vertical lines if provided
    if vlines:
        for vline_date, vline_color, vline_label in vlines:
            ax.axvline(vline_date, color=vline_color, linestyle='--', linewidth=1.5, alpha=0.7, label=vline_label)

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    cum_file = output_dir / f'{file_prefix}cumulative.png'
    plt.savefig(cum_file, dpi=150, facecolor='#0d1117')
    print(f"  Cumulative: {cum_file}")
    plt.close()

    # Plot 2: P-Value Histogram
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(perm_pfs, bins=50, color='steelblue', alpha=0.7, edgecolor='white', label='Permutations')
    ax.axvline(real_pf, color='red', linestyle='--', linewidth=2.5, label=f'Real PF: {real_pf:.4f}')
    ax.axvline(np.mean(perm_pfs), color='yellow', linestyle=':', linewidth=2, alpha=0.7, label=f'Mean: {np.mean(perm_pfs):.4f}')

    ax.set_xlabel("Profit Factor")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{ticker} {prefix.title()} MCPT ({strategy}) | P-Value: {p_value:.4f}")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    mcpt_file = output_dir / f'{file_prefix}mcpt.png'
    plt.savefig(mcpt_file, dpi=150, facecolor='#0d1117')
    print(f"  MCPT Histogram: {mcpt_file}")
    plt.close()

    return [cum_file, mcpt_file]
