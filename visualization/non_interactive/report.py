#!/usr/bin/env python3
"""
Markdown Report Generator for Backtest Results

Generates comprehensive markdown reports with all 12 statistical blocks:
1. Performance
2. Risk
3. Volatility
4. Risk-Return Ratios
5. Trade Statistics
6. PnL Distribution
7. Statistical Robustness
8. Temporal Stability
9. Dependencies
10. Costs (TODO)
11. Concentration
12. Overfitting (TODO)

Supports: MCPT (insample, walkforward), Bootstrap (ticker, batch)
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _indicator(value: float, threshold: float, higher_is_better: bool = True) -> str:
    """Return indicator symbol based on value vs threshold."""
    if value is None:
        return " "
    if higher_is_better:
        return "+" if value >= threshold else "-"
    return "+" if value <= threshold else "-"


def _format_value(value: Any, decimals: int = 4) -> str:
    """Format value for display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.{decimals}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _section_header(title: str, level: int = 2) -> List[str]:
    """Generate section header."""
    prefix = "#" * level
    return [f"{prefix} {title}", ""]


def _metric_table(rows: List[tuple], headers: List[str] = None) -> List[str]:
    """Generate metric table with headers."""
    if headers is None:
        headers = ["Metric", "Value", "Status"]

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["--------"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.append("")
    return lines


# ============================================================================
# BLOCK GENERATORS
# ============================================================================

def _block_performance(stats: Dict) -> List[str]:
    """Block 1: Performance metrics."""
    perf = stats.get('performance', {})
    if not perf:
        return []

    lines = _section_header("Performance", 3)
    rows = [
        ("Total Return", f"{_format_value(perf.get('total_return_pct', 0), 2)}%",
         _indicator(perf.get('total_return_pct', 0), 0)),
        ("Cumulative Log Return", _format_value(perf.get('cumulative_log_return', 0)),
         _indicator(perf.get('cumulative_log_return', 0), 0)),
        ("CAGR", f"{_format_value(perf.get('cagr_pct', 0), 2)}%",
         _indicator(perf.get('cagr_pct', 0), 0)),
        ("Total PnL", _format_value(perf.get('total_pnl', 0)),
         _indicator(perf.get('total_pnl', 0), 0)),
        ("Mean Return/Period", _format_value(perf.get('mean_return_period', 0), 6), ""),
        ("Mean Return/Trade", _format_value(perf.get('mean_return_trade', 0), 6), ""),
    ]
    lines.extend(_metric_table(rows))
    return lines


def _block_risk(stats: Dict) -> List[str]:
    """Block 2: Risk metrics."""
    risk = stats.get('risk', {})
    if not risk:
        return []

    lines = _section_header("Risk", 3)
    rows = [
        ("Max Drawdown", f"{_format_value(risk.get('max_dd_pct', 0), 2)}%",
         _indicator(abs(risk.get('max_dd_pct', 0)), 20, False)),
        ("Max DD Absolute", _format_value(risk.get('max_dd_abs', 0)), ""),
        ("Max DD Duration", f"{risk.get('max_dd_duration', 0):,} bars",
         _indicator(risk.get('max_dd_duration', 0), 500, False)),
        ("Ulcer Index", _format_value(risk.get('ulcer_index', 0)),
         _indicator(risk.get('ulcer_index', 0), 5, False)),
        ("Mean Drawdown", f"{_format_value(risk.get('mean_dd', 0), 2)}%", ""),
    ]
    lines.extend(_metric_table(rows))
    return lines


def _block_volatility(stats: Dict) -> List[str]:
    """Block 3: Volatility metrics."""
    vol = stats.get('volatility', {})
    if not vol:
        return []

    lines = _section_header("Volatility", 3)
    rows = [
        ("Annualized Volatility", f"{_format_value(vol.get('annualized_vol_pct', 0), 2)}%", ""),
        ("Monthly Volatility", f"{_format_value(vol.get('monthly_vol_pct', 0), 2)}%", ""),
        ("Downside Volatility", f"{_format_value(vol.get('downside_vol_pct', 0), 2)}%",
         _indicator(vol.get('downside_vol_pct', 0), vol.get('annualized_vol_pct', 0), False)),
        ("Semi-Deviation", _format_value(vol.get('semi_deviation', 0), 6), ""),
    ]
    lines.extend(_metric_table(rows))
    return lines


def _block_risk_return(stats: Dict) -> List[str]:
    """Block 4: Risk-Return Ratios."""
    rr = stats.get('risk_return', {})
    if not rr:
        return []

    lines = _section_header("Risk-Return Ratios", 3)
    rows = [
        ("Sharpe Ratio", _format_value(rr.get('sharpe_ratio', 0)),
         _indicator(rr.get('sharpe_ratio', 0), 0.5)),
        ("Sortino Ratio", _format_value(rr.get('sortino_ratio', 0)),
         _indicator(rr.get('sortino_ratio', 0), 0.5)),
        ("Calmar Ratio", _format_value(rr.get('calmar_ratio', 0)),
         _indicator(rr.get('calmar_ratio', 0), 0.5)),
        ("SQN", _format_value(rr.get('sqn', 0)),
         _indicator(rr.get('sqn', 0), 1.5)),
    ]
    lines.extend(_metric_table(rows))
    return lines


def _block_trade_stats(stats: Dict) -> List[str]:
    """Block 5: Trade Statistics."""
    trades = stats.get('trade_stats', {})
    if not trades:
        return []

    lines = _section_header("Trade Statistics", 3)
    rows = [
        ("Number of Trades", f"{trades.get('n_trades', 0):,}",
         _indicator(trades.get('n_trades', 0), 30)),
        ("Win Rate", f"{_format_value(trades.get('win_rate_pct', 0), 2)}%",
         _indicator(trades.get('win_rate_pct', 0), 50)),
        ("Average Gain", _format_value(trades.get('avg_gain', 0), 6), ""),
        ("Average Loss", _format_value(trades.get('avg_loss', 0), 6), ""),
        ("Profit Factor", _format_value(trades.get('profit_factor', 0)),
         _indicator(trades.get('profit_factor', 0), 1.0)),
        ("Expectancy", _format_value(trades.get('expectancy', 0), 6),
         _indicator(trades.get('expectancy', 0), 0)),
    ]
    lines.extend(_metric_table(rows))
    return lines


def _block_distribution(stats: Dict) -> List[str]:
    """Block 6: PnL Distribution."""
    dist = stats.get('distribution', {})
    if not dist:
        return []

    lines = _section_header("PnL Distribution", 3)
    rows = [
        ("Skewness", _format_value(dist.get('skewness', 0)),
         _indicator(dist.get('skewness', 0), 0)),
        ("Kurtosis", _format_value(dist.get('kurtosis', 0)), ""),
        ("P5", _format_value(dist.get('p5', 0), 6), ""),
        ("P25", _format_value(dist.get('p25', 0), 6), ""),
        ("Median (P50)", _format_value(dist.get('p50', 0), 6), ""),
        ("P75", _format_value(dist.get('p75', 0), 6), ""),
        ("P95", _format_value(dist.get('p95', 0), 6), ""),
        ("Best Trade", _format_value(dist.get('best_trade', 0), 6), ""),
        ("Worst Trade", _format_value(dist.get('worst_trade', 0), 6), ""),
    ]
    lines.extend(_metric_table(rows))
    return lines


def _block_robustness(stats: Dict, analysis_type: str = 'mcpt') -> List[str]:
    """Block 7: Statistical Robustness."""
    rob = stats.get('robustness', {})
    if not rob:
        return []

    lines = _section_header("Statistical Robustness", 3)

    p_value = rob.get('p_value', 1.0)
    significant = p_value < 0.05

    rows = [
        ("P-Value", _format_value(p_value),
         _indicator(p_value, 0.05, False)),
        ("Significant (p<0.05)", "Yes" if significant else "No",
         "+" if significant else "-"),
        ("Percentile vs Perms", f"{_format_value(rob.get('percentile_vs_perms', 0), 2)}%", ""),
        ("Viable Strategies Ratio", f"{_format_value(rob.get('viable_strategies_ratio', 0) * 100, 2)}%", ""),
    ]

    if rob.get('pbo') is not None:
        rows.append(("PBO", _format_value(rob.get('pbo')), ""))

    lines.extend(_metric_table(rows))
    return lines


def _block_temporal(stats: Dict) -> List[str]:
    """Block 8: Temporal Stability."""
    temp = stats.get('temporal', {})
    if not temp:
        return []

    yearly = temp.get('yearly_stats', {})
    variability = temp.get('variability', {})

    if not yearly:
        return []

    lines = _section_header("Temporal Stability (by Year)", 3)

    # Yearly table
    headers = ["Year", "Cum Log", "PnL", "CAGR%", "MaxDD%", "Sharpe", "SQN"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["------"] * len(headers)) + "|")

    for year in sorted(yearly.keys()):
        y = yearly[year]
        lines.append(
            f"| {year} | {_format_value(y.get('cumulative_log', 0))} | "
            f"{_format_value(y.get('pnl', 0))} | {_format_value(y.get('cagr_pct', 0), 2)} | "
            f"{_format_value(y.get('max_dd_pct', 0), 2)} | {_format_value(y.get('sharpe', 0))} | "
            f"{_format_value(y.get('sqn', 0))} |"
        )

    lines.append("")

    # Variability
    if variability:
        lines.extend(_section_header("Year-to-Year Variability", 4))
        var_rows = []
        for key, val in variability.items():
            metric_name = key.replace('_std', ' (Std)').replace('_', ' ').title()
            var_rows.append((metric_name, _format_value(val), ""))
        lines.extend(_metric_table(var_rows))

    return lines


def _block_dependencies(stats: Dict) -> List[str]:
    """Block 9: Dependencies."""
    deps = stats.get('dependencies', {})
    if not deps:
        return []

    lines = _section_header("Dependencies", 3)
    rows = [
        ("Return Autocorr (lag=1)", _format_value(deps.get('return_autocorr_1', 0)), ""),
        ("Return Autocorr (lag=5)", _format_value(deps.get('return_autocorr_5', 0)), ""),
        ("DD Autocorr (lag=1)", _format_value(deps.get('dd_autocorr_1', 0)), ""),
        ("Hurst Exponent", _format_value(deps.get('hurst_exponent', 0.5)),
         _indicator(abs(deps.get('hurst_exponent', 0.5) - 0.5), 0.1, False)),
    ]

    if deps.get('tail_dependence') is not None:
        rows.append(("Tail Dependence", _format_value(deps.get('tail_dependence')), ""))

    lines.extend(_metric_table(rows))
    return lines


def _block_concentration(stats: Dict) -> List[str]:
    """Block 11: Concentration."""
    conc = stats.get('concentration', {})
    if not conc:
        return []

    lines = _section_header("Concentration", 3)
    rows = [
        ("Gini Coefficient", _format_value(conc.get('gini', 0)),
         _indicator(conc.get('gini', 0), 0.5, False)),
        ("Top 5% Trades PnL Share", f"{_format_value(conc.get('top_5pct_pnl_share', 0), 2)}%",
         _indicator(conc.get('top_5pct_pnl_share', 0), 50, False)),
    ]

    if conc.get('risk_per_asset') is not None:
        rows.append(("Risk per Asset", _format_value(conc.get('risk_per_asset')), ""))
    if conc.get('risk_per_period') is not None:
        rows.append(("Risk per Period", _format_value(conc.get('risk_per_period')), ""))

    lines.extend(_metric_table(rows))
    return lines


# ============================================================================
# TICKER REPORT GENERATORS
# ============================================================================

def generate_mcpt_ticker_report(results: Dict, output_path: Path,
                                 report_type: str = 'insample') -> Path:
    """
    Generate comprehensive markdown report for MCPT single ticker.

    Args:
        results: Results dict from MCPT analysis
        output_path: Directory to save report
        report_type: 'insample' or 'walkforward'

    Returns:
        Path to generated report
    """
    output_path.mkdir(parents=True, exist_ok=True)

    ticker = results.get('ticker', 'Unknown')
    strategy = results.get('strategy', 'Unknown')

    report_file = output_path / f'{ticker}_{report_type}_report.md'

    lines = [
        f"# {ticker} - {strategy.upper()} MCPT {report_type.title()} Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Analysis Type:** Monte Carlo Permutation Test ({report_type})",
        "",
        "---",
        "",
    ]

    # Summary section
    lines.extend(_section_header("Summary"))

    p_value = results.get('p_value', 1.0)
    real_pf = results.get('real_pf', 0)
    significant = p_value < 0.05
    n_perms = results.get('n_permutations', 0)

    summary_rows = [
        ("P-Value", _format_value(p_value), _indicator(p_value, 0.05, False)),
        ("Profit Factor", _format_value(real_pf), _indicator(real_pf, 1.0)),
        ("Significant (p<0.05)", "Yes" if significant else "No", "+" if significant else "-"),
        ("Permutations", f"{n_perms:,}", ""),
    ]
    lines.extend(_metric_table(summary_rows))

    # Period info
    period = results.get('period', {})
    lines.extend(_section_header("Period"))
    lines.extend([
        f"- **Start:** {period.get('start', 'N/A')}",
        f"- **End:** {period.get('end', 'N/A')}",
        f"- **Candles:** {results.get('n_candles', 0):,}",
        "",
    ])

    # All 12 blocks
    stats = results.get('stats', {})

    lines.extend(_section_header("Detailed Statistics"))
    lines.extend(_block_performance(stats))
    lines.extend(_block_risk(stats))
    lines.extend(_block_volatility(stats))
    lines.extend(_block_risk_return(stats))
    lines.extend(_block_trade_stats(stats))
    lines.extend(_block_distribution(stats))
    lines.extend(_block_robustness(stats, 'mcpt'))
    lines.extend(_block_temporal(stats))
    lines.extend(_block_dependencies(stats))
    lines.extend(_block_concentration(stats))

    # Footer
    lines.extend([
        "---",
        "",
        f"*Report generated by Trading-System MCPT ({report_type})*",
    ])

    report_file.write_text('\n'.join(lines))
    return report_file


def generate_bootstrap_ticker_report(results: Dict, output_path: Path,
                                      bootstrap_type: str = 'circular_block') -> Path:
    """
    Generate comprehensive markdown report for Bootstrap single ticker.

    Args:
        results: Results dict from Bootstrap analysis
        output_path: Directory to save report
        bootstrap_type: 'circular_block', 'stationary', or 'trade_based'

    Returns:
        Path to generated report
    """
    output_path.mkdir(parents=True, exist_ok=True)

    ticker = results.get('ticker', 'Unknown')
    strategy = results.get('strategy', 'Unknown')

    report_file = output_path / f'{ticker}_bootstrap_{bootstrap_type}_report.md'

    bootstrap_name = bootstrap_type.replace('_', ' ').title()

    lines = [
        f"# {ticker} - {strategy.upper()} {bootstrap_name} Bootstrap Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Analysis Type:** {bootstrap_name} Bootstrap",
        "",
        "---",
        "",
    ]

    # Summary section
    lines.extend(_section_header("Summary"))

    p_value = results.get('p_value', 1.0)
    real_metric = results.get('real_metric', results.get('real_pf', 0))
    significant = p_value < 0.05
    n_iterations = results.get('n_iterations', 0)
    ci_95 = results.get('ci_95', None)

    summary_rows = [
        ("P-Value", _format_value(p_value), _indicator(p_value, 0.05, False)),
        ("Real Metric (PF)", _format_value(real_metric), _indicator(real_metric, 1.0)),
        ("Significant (p<0.05)", "Yes" if significant else "No", "+" if significant else "-"),
        ("Bootstrap Iterations", f"{n_iterations:,}", ""),
    ]

    if ci_95:
        summary_rows.append(("95% CI", f"[{_format_value(ci_95[0])}, {_format_value(ci_95[1])}]", ""))

    lines.extend(_metric_table(summary_rows))

    # Bootstrap parameters
    lines.extend(_section_header("Bootstrap Parameters"))
    params = results.get('params', {})
    if params:
        param_rows = []
        for key, val in params.items():
            param_rows.append((key.replace('_', ' ').title(), str(val), ""))
        lines.extend(_metric_table(param_rows, ["Parameter", "Value", ""]))

    # Period info
    period = results.get('period', {})
    lines.extend(_section_header("Period"))
    lines.extend([
        f"- **Start:** {period.get('start', 'N/A')}",
        f"- **End:** {period.get('end', 'N/A')}",
        f"- **Candles:** {results.get('n_candles', 0):,}",
        "",
    ])

    # All 12 blocks
    stats = results.get('stats', {})

    lines.extend(_section_header("Detailed Statistics"))
    lines.extend(_block_performance(stats))
    lines.extend(_block_risk(stats))
    lines.extend(_block_volatility(stats))
    lines.extend(_block_risk_return(stats))
    lines.extend(_block_trade_stats(stats))
    lines.extend(_block_distribution(stats))
    lines.extend(_block_robustness(stats, 'bootstrap'))
    lines.extend(_block_temporal(stats))
    lines.extend(_block_dependencies(stats))
    lines.extend(_block_concentration(stats))

    # Footer
    lines.extend([
        "---",
        "",
        f"*Report generated by Trading-System Bootstrap ({bootstrap_type})*",
    ])

    report_file.write_text('\n'.join(lines))
    return report_file


# ============================================================================
# BATCH REPORT GENERATORS
# ============================================================================

def generate_mcpt_batch_report(batch_name: str, strategy: str,
                                ticker_results: List[Dict], batch_stats: Dict,
                                output_path: Path, report_type: str = 'insample') -> Path:
    """
    Generate comprehensive markdown report for MCPT batch.

    Args:
        batch_name: Name of the batch
        strategy: Strategy name
        ticker_results: List of individual ticker results
        batch_stats: Aggregated batch statistics
        output_path: Directory to save report
        report_type: 'insample' or 'walkforward'

    Returns:
        Path to generated report
    """
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / f'{batch_name}_{report_type}_report.md'

    lines = [
        f"# {batch_name} - {strategy.upper()} MCPT {report_type.title()} Batch Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Analysis Type:** Monte Carlo Permutation Test ({report_type})",
        "",
        "---",
        "",
    ]

    # Summary
    lines.extend(_section_header("Batch Summary"))

    n_tickers = len(ticker_results)
    n_significant = sum(1 for r in ticker_results if r.get('p_value', 1) < 0.05)
    pct_significant = (100 * n_significant / n_tickers) if n_tickers else 0

    summary_rows = [
        ("Total Tickers", str(n_tickers), ""),
        ("Significant (p<0.05)", f"{n_significant} ({pct_significant:.1f}%)",
         _indicator(pct_significant, 50)),
        ("Strategy", strategy.upper(), ""),
    ]
    lines.extend(_metric_table(summary_rows))

    # Ticker Results Table
    lines.extend(_section_header("Results by Ticker"))

    headers = ["Ticker", "P-Value", "PF", "CAGR%", "MaxDD%", "Sharpe", "SQN", "Status"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["------"] * len(headers)) + "|")

    # Sort by p-value
    sorted_results = sorted(ticker_results, key=lambda x: x.get('p_value', 1))

    for r in sorted_results:
        ticker = r.get('ticker', '?')
        p_val = r.get('p_value', 1)
        stats = r.get('stats', {})
        perf = stats.get('performance', {})
        risk = stats.get('risk', {})
        rr = stats.get('risk_return', {})
        trade = stats.get('trade_stats', {})

        status = "+" if p_val < 0.05 else "-"

        lines.append(
            f"| {ticker} | {_format_value(p_val)} | "
            f"{_format_value(trade.get('profit_factor', 0))} | "
            f"{_format_value(perf.get('cagr_pct', 0), 2)} | "
            f"{_format_value(risk.get('max_dd_pct', 0), 2)} | "
            f"{_format_value(rr.get('sharpe_ratio', 0))} | "
            f"{_format_value(rr.get('sqn', 0))} | {status} |"
        )

    lines.append("")

    # Aggregate Statistics
    lines.extend(_section_header("Aggregate Statistics"))
    _add_batch_aggregate_tables(lines, batch_stats)

    # Footer
    lines.extend([
        "---",
        "",
        f"*Report generated by Trading-System MCPT Batch ({report_type})*",
    ])

    report_file.write_text('\n'.join(lines))
    return report_file


def generate_bootstrap_batch_report(batch_name: str, strategy: str,
                                     ticker_results: List[Dict], batch_stats: Dict,
                                     output_path: Path,
                                     bootstrap_type: str = 'circular_block') -> Path:
    """
    Generate comprehensive markdown report for Bootstrap batch.

    Args:
        batch_name: Name of the batch
        strategy: Strategy name
        ticker_results: List of individual ticker results
        batch_stats: Aggregated batch statistics
        output_path: Directory to save report
        bootstrap_type: 'circular_block', 'stationary', or 'trade_based'

    Returns:
        Path to generated report
    """
    output_path.mkdir(parents=True, exist_ok=True)

    bootstrap_name = bootstrap_type.replace('_', ' ').title()
    report_file = output_path / f'{batch_name}_bootstrap_{bootstrap_type}_report.md'

    lines = [
        f"# {batch_name} - {strategy.upper()} {bootstrap_name} Bootstrap Batch Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Analysis Type:** {bootstrap_name} Bootstrap",
        "",
        "---",
        "",
    ]

    # Summary
    lines.extend(_section_header("Batch Summary"))

    n_tickers = len(ticker_results)
    n_significant = sum(1 for r in ticker_results if r.get('p_value', 1) < 0.05)
    pct_significant = (100 * n_significant / n_tickers) if n_tickers else 0

    summary_rows = [
        ("Total Tickers", str(n_tickers), ""),
        ("Significant (p<0.05)", f"{n_significant} ({pct_significant:.1f}%)",
         _indicator(pct_significant, 50)),
        ("Strategy", strategy.upper(), ""),
        ("Bootstrap Type", bootstrap_name, ""),
    ]
    lines.extend(_metric_table(summary_rows))

    # Ticker Results Table
    lines.extend(_section_header("Results by Ticker"))

    headers = ["Ticker", "P-Value", "Real Metric", "CI Lower", "CI Upper", "Status"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["------"] * len(headers)) + "|")

    # Sort by p-value
    sorted_results = sorted(ticker_results, key=lambda x: x.get('p_value', 1))

    for r in sorted_results:
        ticker = r.get('ticker', '?')
        p_val = r.get('p_value', 1)
        real_metric = r.get('real_metric', r.get('real_pf', 0))
        ci_95 = r.get('ci_95', (None, None))

        status = "+" if p_val < 0.05 else "-"
        ci_low = _format_value(ci_95[0]) if ci_95[0] is not None else "N/A"
        ci_high = _format_value(ci_95[1]) if ci_95[1] is not None else "N/A"

        lines.append(
            f"| {ticker} | {_format_value(p_val)} | "
            f"{_format_value(real_metric)} | "
            f"{ci_low} | {ci_high} | {status} |"
        )

    lines.append("")

    # Extended results table with all stats
    lines.extend(_section_header("Extended Results"))

    headers2 = ["Ticker", "PF", "CAGR%", "MaxDD%", "Sharpe", "SQN", "Trades"]
    lines.append("| " + " | ".join(headers2) + " |")
    lines.append("|" + "|".join(["------"] * len(headers2)) + "|")

    for r in sorted_results:
        ticker = r.get('ticker', '?')
        stats = r.get('stats', {})
        perf = stats.get('performance', {})
        risk = stats.get('risk', {})
        rr = stats.get('risk_return', {})
        trade = stats.get('trade_stats', {})

        lines.append(
            f"| {ticker} | {_format_value(trade.get('profit_factor', 0))} | "
            f"{_format_value(perf.get('cagr_pct', 0), 2)} | "
            f"{_format_value(risk.get('max_dd_pct', 0), 2)} | "
            f"{_format_value(rr.get('sharpe_ratio', 0))} | "
            f"{_format_value(rr.get('sqn', 0))} | "
            f"{trade.get('n_trades', 0)} |"
        )

    lines.append("")

    # Aggregate Statistics
    lines.extend(_section_header("Aggregate Statistics"))
    _add_batch_aggregate_tables(lines, batch_stats)

    # Footer
    lines.extend([
        "---",
        "",
        f"*Report generated by Trading-System Bootstrap Batch ({bootstrap_type})*",
    ])

    report_file.write_text('\n'.join(lines))
    return report_file


def _add_batch_aggregate_tables(lines: List[str], batch_stats: Dict) -> None:
    """Add aggregate statistics tables for batch reports (all 12 blocks)."""

    # Complete mapping of all 12 blocks with ALL metrics
    metric_groups = {
        # Block 1: Performance
        'Performance': [
            ('performance.total_return_pct', 'Total Return %'),
            ('performance.cumulative_log_return', 'Cumulative Log Return'),
            ('performance.cagr_pct', 'CAGR %'),
            ('performance.total_pnl', 'Total PnL'),
            ('performance.mean_return_period', 'Mean Return/Period'),
            ('performance.mean_return_trade', 'Mean Return/Trade'),
        ],
        # Block 2: Risk
        'Risk': [
            ('risk.max_dd_pct', 'Max Drawdown %'),
            ('risk.max_dd_abs', 'Max DD Absolute'),
            ('risk.max_dd_duration', 'Max DD Duration'),
            ('risk.ulcer_index', 'Ulcer Index'),
            ('risk.mean_dd', 'Mean Drawdown'),
        ],
        # Block 3: Volatility
        'Volatility': [
            ('volatility.annualized_vol_pct', 'Annualized Vol %'),
            ('volatility.monthly_vol_pct', 'Monthly Vol %'),
            ('volatility.downside_vol_pct', 'Downside Vol %'),
            ('volatility.semi_deviation', 'Semi-Deviation'),
        ],
        # Block 4: Risk-Return Ratios
        'Risk-Return Ratios': [
            ('risk_return.sharpe_ratio', 'Sharpe Ratio'),
            ('risk_return.sortino_ratio', 'Sortino Ratio'),
            ('risk_return.calmar_ratio', 'Calmar Ratio'),
            ('risk_return.sqn', 'SQN'),
        ],
        # Block 5: Trade Statistics
        'Trade Statistics': [
            ('trade_stats.n_trades', 'N Trades'),
            ('trade_stats.win_rate_pct', 'Win Rate %'),
            ('trade_stats.avg_gain', 'Average Gain'),
            ('trade_stats.avg_loss', 'Average Loss'),
            ('trade_stats.profit_factor', 'Profit Factor'),
            ('trade_stats.expectancy', 'Expectancy'),
        ],
        # Block 6: PnL Distribution
        'PnL Distribution': [
            ('distribution.skewness', 'Skewness'),
            ('distribution.kurtosis', 'Kurtosis'),
            ('distribution.p5', 'Percentile 5'),
            ('distribution.p25', 'Percentile 25'),
            ('distribution.p50', 'Median (P50)'),
            ('distribution.p75', 'Percentile 75'),
            ('distribution.p95', 'Percentile 95'),
            ('distribution.best_trade', 'Best Trade'),
            ('distribution.worst_trade', 'Worst Trade'),
        ],
        # Block 7: Statistical Robustness
        'Statistical Robustness': [
            ('robustness.p_value', 'P-Value'),
            ('robustness.percentile_vs_perms', 'Percentile vs Perms'),
            ('robustness.viable_strategies_ratio', 'Viable Strategies Ratio'),
        ],
        # Block 9: Dependencies
        'Dependencies': [
            ('dependencies.return_autocorr_1', 'Return Autocorr (lag=1)'),
            ('dependencies.return_autocorr_5', 'Return Autocorr (lag=5)'),
            ('dependencies.dd_autocorr_1', 'DD Autocorr (lag=1)'),
            ('dependencies.hurst_exponent', 'Hurst Exponent'),
        ],
        # Block 11: Concentration
        'Concentration': [
            ('concentration.gini', 'Gini Coefficient'),
            ('concentration.top_5pct_pnl_share', 'Top 5% PnL Share'),
        ],
    }

    for group_name, metrics in metric_groups.items():
        # Check if any metric in this group has data
        has_data = any(key in batch_stats for key, _ in metrics)
        if not has_data:
            continue

        lines.extend(_section_header(group_name, 3))

        headers = ["Metric", "Mean", "Median", "Std", "Min", "Max"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["------"] * len(headers)) + "|")

        for key, label in metrics:
            if key in batch_stats:
                s = batch_stats[key]
                lines.append(
                    f"| {label} | {_format_value(s.get('mean', 0))} | "
                    f"{_format_value(s.get('median', 0))} | "
                    f"{_format_value(s.get('std', 0))} | "
                    f"{_format_value(s.get('min', 0))} | "
                    f"{_format_value(s.get('max', 0))} |"
                )

        lines.append("")
