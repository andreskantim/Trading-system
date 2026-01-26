#!/usr/bin/env python3
"""
Comprehensive Statistics Calculator for Backtest Results

Implements 12 blocks of trading statistics:
- Block 1: Performance (Total Return, Cumulative log return, CAGR, PnL)
- Block 2: Risk/Drawdowns (Max DD %, duration, Ulcer Index)
- Block 3: Volatility (Annualized, Monthly, Downside)
- Block 4: Risk-Return Ratios (Sharpe, Sortino, Calmar, SQN)
- Block 5: Trade Statistics (Win rate, Profit Factor, Expectancy)
- Block 6: PnL Distribution (Skewness, Kurtosis, Percentiles)
- Block 7: Statistical Robustness (p-value, PBO)
- Block 8: Temporal Stability (Rolling Sharpe/CAGR/DD)
- Block 9: Dependencies (Autocorrelation, Hurst exponent)
- Block 10: Costs (COMMENTED - not implemented)
- Block 11: Concentration Risk (Gini, % PnL in top trades)
- Block 12: Overfitting Diagnostics (COMMENTED - not implemented)

Usage:
    from utils.stats_calculator import calculate_all_stats, calculate_batch_stats

    stats = calculate_all_stats(returns, signal, prices)
    batch_stats = calculate_batch_stats(list_of_results)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats as scipy_stats


# ============================================================================
# BLOCK 1: PERFORMANCE METRICS
# ============================================================================

def calc_total_return(cum_log_returns: np.ndarray) -> float:
    """Calculate total return as percentage."""
    if len(cum_log_returns) == 0:
        return 0.0
    return float(np.exp(cum_log_returns[-1]) - 1) * 100


def calc_cumulative_log_return(returns: np.ndarray) -> float:
    """Calculate cumulative log return."""
    return float(np.nansum(returns))


def calc_cagr(cum_log_returns: np.ndarray, n_periods: int, periods_per_year: int = 8760) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        cum_log_returns: Cumulative log returns
        n_periods: Number of periods (bars)
        periods_per_year: Number of periods per year (8760 for hourly data)
    """
    if n_periods == 0:
        return 0.0
    years = n_periods / periods_per_year
    if years == 0:
        return 0.0
    total_return = np.exp(cum_log_returns[-1]) if len(cum_log_returns) > 0 else 1.0
    return float((total_return ** (1 / years) - 1) * 100)


def calc_pnl(returns: np.ndarray) -> float:
    """Calculate total PnL in log return units."""
    return float(np.nansum(returns))


def calc_performance_block(returns: np.ndarray, periods_per_year: int = 8760) -> Dict[str, float]:
    """Calculate Block 1: Performance metrics."""
    returns = np.array(returns)
    cum_rets = np.nancumsum(returns)
    n_periods = len(returns)

    return {
        'total_return_pct': calc_total_return(cum_rets),
        'cumulative_log_return': calc_cumulative_log_return(returns),
        'cagr_pct': calc_cagr(cum_rets, n_periods, periods_per_year),
        'pnl': calc_pnl(returns),
    }


# ============================================================================
# BLOCK 2: RISK / DRAWDOWNS
# ============================================================================

def calc_drawdown_series(cum_returns: np.ndarray) -> np.ndarray:
    """Calculate drawdown series from cumulative returns."""
    peak = np.maximum.accumulate(cum_returns)
    drawdown = cum_returns - peak
    return drawdown


def calc_max_drawdown_pct(cum_log_returns: np.ndarray) -> float:
    """Calculate maximum drawdown as percentage."""
    if len(cum_log_returns) == 0:
        return 0.0
    equity = np.exp(cum_log_returns)
    peak = np.maximum.accumulate(equity)
    drawdown_pct = (equity - peak) / peak * 100
    return float(np.nanmin(drawdown_pct))


def calc_max_drawdown_duration(cum_log_returns: np.ndarray) -> int:
    """Calculate maximum drawdown duration in periods."""
    if len(cum_log_returns) == 0:
        return 0
    equity = np.exp(cum_log_returns)
    peak = np.maximum.accumulate(equity)
    in_drawdown = equity < peak

    max_duration = 0
    current_duration = 0

    for i in range(len(in_drawdown)):
        if in_drawdown[i]:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


def calc_ulcer_index(cum_log_returns: np.ndarray) -> float:
    """
    Calculate Ulcer Index (UI) - measures downside risk.
    Lower is better. UI = sqrt(mean(drawdown^2))
    """
    if len(cum_log_returns) == 0:
        return 0.0
    equity = np.exp(cum_log_returns)
    peak = np.maximum.accumulate(equity)
    drawdown_pct = ((equity - peak) / peak) * 100
    return float(np.sqrt(np.mean(drawdown_pct ** 2)))


def calc_risk_block(returns: np.ndarray) -> Dict[str, float]:
    """Calculate Block 2: Risk/Drawdowns metrics."""
    returns = np.array(returns)
    cum_rets = np.nancumsum(returns)

    return {
        'max_drawdown_pct': calc_max_drawdown_pct(cum_rets),
        'max_drawdown_duration': calc_max_drawdown_duration(cum_rets),
        'ulcer_index': calc_ulcer_index(cum_rets),
    }


# ============================================================================
# BLOCK 3: VOLATILITY
# ============================================================================

def calc_annualized_volatility(returns: np.ndarray, periods_per_year: int = 8760) -> float:
    """Calculate annualized volatility."""
    if len(returns) < 2:
        return 0.0
    return float(np.nanstd(returns) * np.sqrt(periods_per_year) * 100)


def calc_monthly_volatility(returns: np.ndarray, periods_per_hour: int = 1) -> float:
    """Calculate monthly volatility (assuming hourly data)."""
    periods_per_month = 30 * 24 * periods_per_hour
    if len(returns) < 2:
        return 0.0
    return float(np.nanstd(returns) * np.sqrt(periods_per_month) * 100)


def calc_downside_volatility(returns: np.ndarray, periods_per_year: int = 8760, threshold: float = 0.0) -> float:
    """Calculate downside volatility (semi-deviation)."""
    negative_returns = returns[returns < threshold]
    if len(negative_returns) < 2:
        return 0.0
    return float(np.nanstd(negative_returns) * np.sqrt(periods_per_year) * 100)


def calc_volatility_block(returns: np.ndarray, periods_per_year: int = 8760) -> Dict[str, float]:
    """Calculate Block 3: Volatility metrics."""
    returns = np.array(returns)

    return {
        'annualized_volatility_pct': calc_annualized_volatility(returns, periods_per_year),
        'monthly_volatility_pct': calc_monthly_volatility(returns),
        'downside_volatility_pct': calc_downside_volatility(returns, periods_per_year),
    }


# ============================================================================
# BLOCK 4: RISK-RETURN RATIOS
# ============================================================================

def calc_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 8760) -> float:
    """
    Calculate annualized Sharpe Ratio.

    Args:
        returns: Array of log returns
        risk_free_rate: Annualized risk-free rate (default 0)
        periods_per_year: Number of periods per year
    """
    if len(returns) < 2:
        return 0.0

    mean_return = np.nanmean(returns)
    std_return = np.nanstd(returns)

    if std_return == 0:
        return 0.0

    # Annualize
    annualized_return = mean_return * periods_per_year
    annualized_std = std_return * np.sqrt(periods_per_year)

    return float((annualized_return - risk_free_rate) / annualized_std)


def calc_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 8760) -> float:
    """
    Calculate annualized Sortino Ratio (uses downside deviation).
    """
    if len(returns) < 2:
        return 0.0

    mean_return = np.nanmean(returns)
    negative_returns = returns[returns < 0]

    if len(negative_returns) < 2:
        return np.inf if mean_return > 0 else 0.0

    downside_std = np.nanstd(negative_returns)

    if downside_std == 0:
        return 0.0

    # Annualize
    annualized_return = mean_return * periods_per_year
    annualized_downside_std = downside_std * np.sqrt(periods_per_year)

    return float((annualized_return - risk_free_rate) / annualized_downside_std)


def calc_calmar_ratio(returns: np.ndarray, periods_per_year: int = 8760) -> float:
    """
    Calculate Calmar Ratio (CAGR / Max Drawdown).
    """
    cum_rets = np.nancumsum(returns)
    n_periods = len(returns)

    cagr = calc_cagr(cum_rets, n_periods, periods_per_year)
    max_dd = abs(calc_max_drawdown_pct(cum_rets))

    if max_dd == 0:
        return 0.0

    return float(cagr / max_dd)


def calc_sqn(returns: np.ndarray) -> float:
    """
    Calculate System Quality Number (SQN).
    SQN = sqrt(N) * mean(R) / std(R)

    SQN < 1.6: Poor
    1.6-2.0: Below average
    2.0-2.5: Average
    2.5-3.0: Good
    3.0-5.0: Excellent
    5.0-7.0: Superb
    > 7.0: Holy Grail
    """
    if len(returns) < 30:
        return 0.0

    mean_r = np.nanmean(returns)
    std_r = np.nanstd(returns)

    if std_r == 0:
        return 0.0

    return float(np.sqrt(len(returns)) * mean_r / std_r)


def calc_risk_return_block(returns: np.ndarray, periods_per_year: int = 8760) -> Dict[str, float]:
    """Calculate Block 4: Risk-Return Ratios."""
    returns = np.array(returns)

    return {
        'sharpe_ratio': calc_sharpe_ratio(returns, 0.0, periods_per_year),
        'sortino_ratio': calc_sortino_ratio(returns, 0.0, periods_per_year),
        'calmar_ratio': calc_calmar_ratio(returns, periods_per_year),
        'sqn': calc_sqn(returns),
    }


# ============================================================================
# BLOCK 5: TRADE STATISTICS
# ============================================================================

def identify_trades(signal: np.ndarray, returns: np.ndarray) -> List[float]:
    """
    Identify individual trades and their PnL.
    A trade starts when signal changes and ends when signal changes again or becomes 0.
    """
    trades = []
    current_trade_pnl = 0.0
    in_trade = False
    prev_signal = 0

    for i in range(len(signal)):
        current_signal = signal[i]

        if current_signal != 0 and not in_trade:
            # Start new trade
            in_trade = True
            current_trade_pnl = returns[i] if not np.isnan(returns[i]) else 0.0
        elif current_signal != 0 and in_trade:
            if current_signal == prev_signal:
                # Continue trade
                current_trade_pnl += returns[i] if not np.isnan(returns[i]) else 0.0
            else:
                # Close current trade, start new one
                trades.append(current_trade_pnl)
                current_trade_pnl = returns[i] if not np.isnan(returns[i]) else 0.0
        elif current_signal == 0 and in_trade:
            # Close trade
            trades.append(current_trade_pnl)
            current_trade_pnl = 0.0
            in_trade = False

        prev_signal = current_signal

    # Close last trade if still open
    if in_trade:
        trades.append(current_trade_pnl)

    return trades


def calc_win_rate(trades: List[float]) -> float:
    """Calculate win rate (percentage of winning trades)."""
    if len(trades) == 0:
        return 0.0
    wins = sum(1 for t in trades if t > 0)
    return float(wins / len(trades) * 100)


def calc_profit_factor(trades: List[float]) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def calc_expectancy(trades: List[float]) -> float:
    """Calculate expectancy (average profit per trade)."""
    if len(trades) == 0:
        return 0.0
    return float(np.mean(trades))


def calc_avg_win_loss_ratio(trades: List[float]) -> float:
    """Calculate average win / average loss ratio."""
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))

    if avg_loss == 0:
        return float('inf')

    return float(avg_win / avg_loss)


def calc_trade_stats_block(signal: np.ndarray, returns: np.ndarray) -> Dict[str, Any]:
    """Calculate Block 5: Trade Statistics."""
    signal = np.array(signal)
    returns = np.array(returns)

    trades = identify_trades(signal, returns)

    return {
        'n_trades': len(trades),
        'win_rate_pct': calc_win_rate(trades),
        'profit_factor': calc_profit_factor(trades),
        'expectancy': calc_expectancy(trades),
        'avg_win_loss_ratio': calc_avg_win_loss_ratio(trades),
    }


# ============================================================================
# BLOCK 6: PnL DISTRIBUTION
# ============================================================================

def calc_skewness(returns: np.ndarray) -> float:
    """Calculate skewness of returns distribution."""
    if len(returns) < 3:
        return 0.0
    return float(scipy_stats.skew(returns[~np.isnan(returns)]))


def calc_kurtosis(returns: np.ndarray) -> float:
    """Calculate excess kurtosis of returns distribution."""
    if len(returns) < 4:
        return 0.0
    return float(scipy_stats.kurtosis(returns[~np.isnan(returns)]))


def calc_percentiles(returns: np.ndarray) -> Dict[str, float]:
    """Calculate key percentiles of returns distribution."""
    clean_returns = returns[~np.isnan(returns)]
    if len(clean_returns) == 0:
        return {'p5': 0.0, 'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'p95': 0.0}

    return {
        'p5': float(np.percentile(clean_returns, 5)),
        'p25': float(np.percentile(clean_returns, 25)),
        'p50': float(np.percentile(clean_returns, 50)),
        'p75': float(np.percentile(clean_returns, 75)),
        'p95': float(np.percentile(clean_returns, 95)),
    }


def calc_distribution_block(returns: np.ndarray) -> Dict[str, float]:
    """Calculate Block 6: PnL Distribution."""
    returns = np.array(returns)
    percentiles = calc_percentiles(returns)

    return {
        'skewness': calc_skewness(returns),
        'kurtosis': calc_kurtosis(returns),
        **percentiles,
    }


# ============================================================================
# BLOCK 7: STATISTICAL ROBUSTNESS
# ============================================================================

def calc_t_test_pvalue(returns: np.ndarray) -> float:
    """
    Calculate p-value from t-test (H0: mean return = 0).
    """
    clean_returns = returns[~np.isnan(returns)]
    if len(clean_returns) < 3:
        return 1.0

    t_stat, p_value = scipy_stats.ttest_1samp(clean_returns, 0)
    return float(p_value)


def calc_robustness_block(returns: np.ndarray, p_value_mcpt: Optional[float] = None) -> Dict[str, float]:
    """
    Calculate Block 7: Statistical Robustness.

    Args:
        returns: Strategy returns
        p_value_mcpt: Pre-calculated MCPT p-value (optional)
    """
    returns = np.array(returns)

    result = {
        't_test_pvalue': calc_t_test_pvalue(returns),
    }

    if p_value_mcpt is not None:
        result['mcpt_pvalue'] = p_value_mcpt

    return result


# ============================================================================
# BLOCK 8: TEMPORAL STABILITY
# ============================================================================

def calc_rolling_sharpe(returns: np.ndarray, window: int = 720, periods_per_year: int = 8760) -> np.ndarray:
    """
    Calculate rolling Sharpe ratio.

    Args:
        returns: Array of returns
        window: Rolling window size (default 720 = ~1 month for hourly)
        periods_per_year: Periods per year for annualization
    """
    if len(returns) < window:
        return np.array([])

    rolling_sharpe = []
    for i in range(window, len(returns)):
        window_returns = returns[i-window:i]
        sharpe = calc_sharpe_ratio(window_returns, 0.0, periods_per_year)
        rolling_sharpe.append(sharpe)

    return np.array(rolling_sharpe)


def calc_rolling_max_dd(returns: np.ndarray, window: int = 720) -> np.ndarray:
    """
    Calculate rolling maximum drawdown.
    """
    if len(returns) < window:
        return np.array([])

    rolling_dd = []
    for i in range(window, len(returns)):
        window_returns = returns[i-window:i]
        cum_rets = np.nancumsum(window_returns)
        max_dd = calc_max_drawdown_pct(cum_rets)
        rolling_dd.append(max_dd)

    return np.array(rolling_dd)


def calc_temporal_stability_block(returns: np.ndarray, window: int = 720,
                                   periods_per_year: int = 8760) -> Dict[str, Any]:
    """Calculate Block 8: Temporal Stability."""
    returns = np.array(returns)

    rolling_sharpe = calc_rolling_sharpe(returns, window, periods_per_year)
    rolling_dd = calc_rolling_max_dd(returns, window)

    result = {
        'rolling_sharpe_mean': float(np.nanmean(rolling_sharpe)) if len(rolling_sharpe) > 0 else 0.0,
        'rolling_sharpe_std': float(np.nanstd(rolling_sharpe)) if len(rolling_sharpe) > 0 else 0.0,
        'rolling_sharpe_min': float(np.nanmin(rolling_sharpe)) if len(rolling_sharpe) > 0 else 0.0,
        'rolling_dd_mean': float(np.nanmean(rolling_dd)) if len(rolling_dd) > 0 else 0.0,
        'rolling_dd_worst': float(np.nanmin(rolling_dd)) if len(rolling_dd) > 0 else 0.0,
    }

    return result


# ============================================================================
# BLOCK 9: DEPENDENCIES / TIME SERIES PROPERTIES
# ============================================================================

def calc_autocorrelation(returns: np.ndarray, lag: int = 1) -> float:
    """Calculate autocorrelation at given lag."""
    clean_returns = returns[~np.isnan(returns)]
    if len(clean_returns) < lag + 2:
        return 0.0

    return float(np.corrcoef(clean_returns[:-lag], clean_returns[lag:])[0, 1])


def calc_hurst_exponent(returns: np.ndarray, max_lag: int = 100) -> float:
    """
    Calculate Hurst exponent using R/S analysis.

    H < 0.5: Mean reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    """
    clean_returns = returns[~np.isnan(returns)]
    if len(clean_returns) < max_lag * 2:
        return 0.5  # Default to random walk

    lags = range(2, max_lag)
    rs_values = []

    for lag in lags:
        n_subsets = len(clean_returns) // lag
        if n_subsets < 1:
            continue

        rs_subset = []
        for i in range(n_subsets):
            subset = clean_returns[i*lag:(i+1)*lag]
            mean_adj = subset - np.mean(subset)
            cum_dev = np.cumsum(mean_adj)
            r = np.max(cum_dev) - np.min(cum_dev)
            s = np.std(subset)
            if s > 0:
                rs_subset.append(r / s)

        if len(rs_subset) > 0:
            rs_values.append((lag, np.mean(rs_subset)))

    if len(rs_values) < 3:
        return 0.5

    # Linear regression on log-log scale
    lags_log = np.log([x[0] for x in rs_values])
    rs_log = np.log([x[1] for x in rs_values])

    slope, _, _, _, _ = scipy_stats.linregress(lags_log, rs_log)

    return float(slope)


def calc_dependencies_block(returns: np.ndarray) -> Dict[str, float]:
    """Calculate Block 9: Dependencies."""
    returns = np.array(returns)

    return {
        'autocorr_lag1': calc_autocorrelation(returns, 1),
        'autocorr_lag5': calc_autocorrelation(returns, 5),
        'autocorr_lag24': calc_autocorrelation(returns, 24),
        'hurst_exponent': calc_hurst_exponent(returns),
    }


# ============================================================================
# BLOCK 10: COSTS (COMMENTED - NOT IMPLEMENTED)
# ============================================================================

# def calc_costs_block(signal: np.ndarray, prices: np.ndarray,
#                      commission_pct: float = 0.1,
#                      slippage_pct: float = 0.05) -> Dict[str, float]:
#     """
#     Calculate Block 10: Transaction Costs.
#
#     Args:
#         signal: Trading signal
#         prices: Price series
#         commission_pct: Commission per trade as percentage
#         slippage_pct: Slippage as percentage
#     """
#     # Count number of trades (signal changes)
#     signal_changes = np.diff(signal)
#     n_trades = np.sum(np.abs(signal_changes) > 0)
#
#     total_commission = n_trades * commission_pct
#     total_slippage = n_trades * slippage_pct
#
#     return {
#         'n_trades': n_trades,
#         'total_commission_pct': total_commission,
#         'total_slippage_pct': total_slippage,
#         'total_costs_pct': total_commission + total_slippage,
#     }


# ============================================================================
# BLOCK 11: CONCENTRATION RISK
# ============================================================================

def calc_gini_coefficient(trades: List[float]) -> float:
    """
    Calculate Gini coefficient of trade PnL distribution.
    0 = perfect equality, 1 = perfect inequality
    """
    if len(trades) == 0:
        return 0.0

    trades_abs = np.abs(trades)
    sorted_trades = np.sort(trades_abs)
    n = len(sorted_trades)

    cumulative = np.cumsum(sorted_trades)

    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n if cumulative[-1] > 0 else 0.0

    return float(max(0, gini))


def calc_top_trades_contribution(trades: List[float], top_n: int = 5) -> float:
    """Calculate percentage of total PnL from top N trades."""
    if len(trades) == 0:
        return 0.0

    sorted_trades = sorted(trades, key=abs, reverse=True)
    top_trades = sorted_trades[:top_n]

    total_pnl = sum(trades)
    top_pnl = sum(top_trades)

    if total_pnl == 0:
        return 0.0

    return float(abs(top_pnl / total_pnl) * 100)


def calc_concentration_block(signal: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
    """Calculate Block 11: Concentration Risk."""
    signal = np.array(signal)
    returns = np.array(returns)

    trades = identify_trades(signal, returns)

    return {
        'gini_coefficient': calc_gini_coefficient(trades),
        'top_5_trades_pct': calc_top_trades_contribution(trades, 5),
        'top_10_trades_pct': calc_top_trades_contribution(trades, 10),
    }


# ============================================================================
# BLOCK 12: OVERFITTING DIAGNOSTICS (COMMENTED - NOT IMPLEMENTED)
# ============================================================================

# def calc_overfitting_block(train_sharpe: float, test_sharpe: float,
#                            n_params: int, n_combinations_tested: int) -> Dict[str, float]:
#     """
#     Calculate Block 12: Overfitting Diagnostics.
#
#     Args:
#         train_sharpe: Sharpe ratio on training set
#         test_sharpe: Sharpe ratio on test set
#         n_params: Number of strategy parameters
#         n_combinations_tested: Number of parameter combinations tested
#     """
#     # Performance degradation
#     if train_sharpe > 0:
#         degradation = (train_sharpe - test_sharpe) / train_sharpe * 100
#     else:
#         degradation = 0.0
#
#     # Deflated Sharpe Ratio approximation
#     # DSR accounts for multiple testing
#     import math
#     expected_max_sharpe = np.sqrt(2 * np.log(n_combinations_tested))
#     dsr = test_sharpe / expected_max_sharpe if expected_max_sharpe > 0 else 0.0
#
#     return {
#         'train_sharpe': train_sharpe,
#         'test_sharpe': test_sharpe,
#         'performance_degradation_pct': degradation,
#         'n_params': n_params,
#         'n_combinations_tested': n_combinations_tested,
#         'deflated_sharpe_ratio': dsr,
#     }


# ============================================================================
# MAIN CALCULATION FUNCTIONS
# ============================================================================

def calculate_all_stats(returns: np.ndarray,
                        signal: np.ndarray,
                        p_value_mcpt: Optional[float] = None,
                        periods_per_year: int = 8760,
                        rolling_window: int = 720) -> Dict[str, Any]:
    """
    Calculate all statistics for a single backtest result.

    Args:
        returns: Strategy returns (signal * price_returns)
        signal: Trading signal array
        p_value_mcpt: Pre-calculated MCPT p-value
        periods_per_year: Number of periods per year (8760 for hourly)
        rolling_window: Window for rolling statistics

    Returns:
        Dictionary with all statistics organized by block
    """
    returns = np.array(returns)
    signal = np.array(signal)

    stats = {}

    # Block 1: Performance
    stats['performance'] = calc_performance_block(returns, periods_per_year)

    # Block 2: Risk
    stats['risk'] = calc_risk_block(returns)

    # Block 3: Volatility
    stats['volatility'] = calc_volatility_block(returns, periods_per_year)

    # Block 4: Risk-Return Ratios
    stats['risk_return'] = calc_risk_return_block(returns, periods_per_year)

    # Block 5: Trade Statistics
    stats['trades'] = calc_trade_stats_block(signal, returns)

    # Block 6: Distribution
    stats['distribution'] = calc_distribution_block(returns)

    # Block 7: Statistical Robustness
    stats['robustness'] = calc_robustness_block(returns, p_value_mcpt)

    # Block 8: Temporal Stability
    stats['temporal'] = calc_temporal_stability_block(returns, rolling_window, periods_per_year)

    # Block 9: Dependencies
    stats['dependencies'] = calc_dependencies_block(returns)

    # Block 11: Concentration
    stats['concentration'] = calc_concentration_block(signal, returns)

    return stats


def flatten_stats(stats: Dict[str, Any], prefix: str = '') -> Dict[str, float]:
    """
    Flatten nested stats dictionary to single level.

    Args:
        stats: Nested statistics dictionary
        prefix: Prefix for keys

    Returns:
        Flat dictionary with dotted keys
    """
    flat = {}
    for key, value in stats.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_stats(value, new_key))
        else:
            flat[new_key] = value
    return flat


def calculate_batch_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate aggregate statistics for a batch of results.

    Args:
        results: List of individual result dictionaries, each containing 'stats' key

    Returns:
        Dictionary with mean, median, std for each metric
    """
    if len(results) == 0:
        return {}

    # Flatten all stats
    flat_results = []
    for result in results:
        if 'stats' in result:
            flat_results.append(flatten_stats(result['stats']))
        else:
            flat_results.append(flatten_stats(result))

    # Get all unique keys
    all_keys = set()
    for flat in flat_results:
        all_keys.update(flat.keys())

    # Calculate aggregates
    aggregates = {}
    for key in all_keys:
        values = [flat.get(key, np.nan) for flat in flat_results]
        values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]

        if len(values) > 0:
            aggregates[key] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)) if len(values) > 1 else 0.0,
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values),
            }

    return aggregates


def get_key_metrics(stats: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract key metrics from stats for summary display.

    Args:
        stats: Full statistics dictionary

    Returns:
        Dictionary with key metrics
    """
    flat = flatten_stats(stats)

    key_metrics = {
        'Total Return (%)': flat.get('performance.total_return_pct', 0.0),
        'CAGR (%)': flat.get('performance.cagr_pct', 0.0),
        'Sharpe Ratio': flat.get('risk_return.sharpe_ratio', 0.0),
        'Sortino Ratio': flat.get('risk_return.sortino_ratio', 0.0),
        'Max Drawdown (%)': flat.get('risk.max_drawdown_pct', 0.0),
        'Win Rate (%)': flat.get('trades.win_rate_pct', 0.0),
        'Profit Factor': flat.get('trades.profit_factor', 0.0),
        'SQN': flat.get('risk_return.sqn', 0.0),
        'N Trades': flat.get('trades.n_trades', 0),
    }

    return key_metrics
