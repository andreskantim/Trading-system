"""
Markdown Report Generator for Backtest Results

Generates formatted markdown reports with tables, indicators, and improved readability.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


def _indicator(value: float, threshold: float, higher_is_better: bool = True) -> str:
    """Return indicator symbol based on value vs threshold."""
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
    return str(value)


def generate_ticker_report(results: Dict, output_path: Path) -> Path:
    """
    Generate markdown report for single ticker backtest.

    Args:
        results: Results dict from MCPT/bootstrap analysis
        output_path: Directory to save report

    Returns:
        Path to generated report
    """
    output_path.mkdir(parents=True, exist_ok=True)

    ticker = results.get('ticker', 'Unknown')
    strategy = results.get('strategy', 'Unknown')
    prefix = results.get('type', 'insample')

    report_file = output_path / f'{ticker}_{prefix}_report.md'

    lines = [
        f"# {ticker} - {strategy.upper()} {prefix.title()} Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]

    # Summary table
    p_value = results.get('p_value', 1.0)
    real_pf = results.get('real_pf', 0)
    significant = results.get('significant', False)

    lines.extend([
        "| Metric | Value | Status |",
        "|--------|-------|--------|",
        f"| P-Value | {_format_value(p_value)} | {_indicator(p_value, 0.05, False)} |",
        f"| Profit Factor | {_format_value(real_pf)} | {_indicator(real_pf, 1.0)} |",
        f"| Significant | {'Yes' if significant else 'No'} | {'+' if significant else '-'} |",
        f"| Permutations | {results.get('n_permutations', 0):,} | |",
        "",
    ])

    # Period info
    period = results.get('period', {})
    lines.extend([
        "## Period",
        "",
        f"- **Start:** {period.get('start', 'N/A')}",
        f"- **End:** {period.get('end', 'N/A')}",
        f"- **Candles:** {results.get('n_candles', 0):,}",
        "",
    ])

    # Stats sections
    stats = results.get('stats', {})

    if 'performance' in stats:
        perf = stats['performance']
        lines.extend([
            "## Performance",
            "",
            "| Metric | Value | Status |",
            "|--------|-------|--------|",
            f"| Total Return | {_format_value(perf.get('total_return_pct', 0), 2)}% | {_indicator(perf.get('total_return_pct', 0), 0)} |",
            f"| CAGR | {_format_value(perf.get('cagr_pct', 0), 2)}% | {_indicator(perf.get('cagr_pct', 0), 0)} |",
            f"| Cumulative Log | {_format_value(perf.get('cumulative_log_return', 0))} | {_indicator(perf.get('cumulative_log_return', 0), 0)} |",
            "",
        ])

    if 'risk' in stats:
        risk = stats['risk']
        lines.extend([
            "## Risk",
            "",
            "| Metric | Value | Status |",
            "|--------|-------|--------|",
            f"| Max Drawdown | {_format_value(risk.get('max_dd_pct', 0), 2)}% | {_indicator(abs(risk.get('max_dd_pct', 0)), 20, False)} |",
            f"| Max DD Duration | {risk.get('max_dd_duration', 0):,} bars | {_indicator(risk.get('max_dd_duration', 0), 1000, False)} |",
            f"| Ulcer Index | {_format_value(risk.get('ulcer_index', 0))} | |",
            "",
        ])

    if 'risk_return' in stats:
        rr = stats['risk_return']
        lines.extend([
            "## Risk-Return Ratios",
            "",
            "| Metric | Value | Status |",
            "|--------|-------|--------|",
            f"| Sharpe Ratio | {_format_value(rr.get('sharpe_ratio', 0))} | {_indicator(rr.get('sharpe_ratio', 0), 0.5)} |",
            f"| Sortino Ratio | {_format_value(rr.get('sortino_ratio', 0))} | {_indicator(rr.get('sortino_ratio', 0), 0.5)} |",
            f"| Calmar Ratio | {_format_value(rr.get('calmar_ratio', 0))} | {_indicator(rr.get('calmar_ratio', 0), 0.5)} |",
            f"| SQN | {_format_value(rr.get('sqn', 0))} | {_indicator(rr.get('sqn', 0), 1.5)} |",
            "",
        ])

    if 'trade_stats' in stats:
        trades = stats['trade_stats']
        lines.extend([
            "## Trade Statistics",
            "",
            "| Metric | Value | Status |",
            "|--------|-------|--------|",
            f"| N Trades | {trades.get('n_trades', 0):,} | |",
            f"| Win Rate | {_format_value(trades.get('win_rate_pct', 0), 2)}% | {_indicator(trades.get('win_rate_pct', 0), 50)} |",
            f"| Profit Factor | {_format_value(trades.get('profit_factor', 0))} | {_indicator(trades.get('profit_factor', 0), 1.0)} |",
            f"| Expectancy | {_format_value(trades.get('expectancy', 0))} | {_indicator(trades.get('expectancy', 0), 0)} |",
            "",
        ])

    lines.extend([
        "---",
        "",
        f"*Report generated by Trading-System MCPT*",
    ])

    report_file.write_text('\n'.join(lines))
    return report_file


def generate_batch_report(batch_name: str, strategy: str, ticker_results: List[Dict],
                          batch_stats: Dict, output_path: Path) -> Path:
    """
    Generate markdown report for batch backtest.

    Args:
        batch_name: Name of the batch
        strategy: Strategy name
        ticker_results: List of individual ticker results
        batch_stats: Aggregated batch statistics
        output_path: Directory to save report

    Returns:
        Path to generated report
    """
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / f'{batch_name}_report.md'

    lines = [
        f"# {batch_name} - {strategy.upper()} Batch Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]

    # Count significant
    n_tickers = len(ticker_results)
    n_significant = sum(1 for r in ticker_results if r.get('p_value', 1) < 0.05)

    lines.extend([
        f"- **Total Tickers:** {n_tickers}",
        f"- **Significant (p<0.05):** {n_significant} ({100*n_significant/n_tickers:.1f}%)" if n_tickers else "",
        "",
    ])

    # Ticker results table
    lines.extend([
        "## Results by Ticker",
        "",
        "| Ticker | P-Value | PF | CAGR% | MaxDD% | Sharpe | Status |",
        "|--------|---------|-----|-------|--------|--------|--------|",
    ])

    # Sort by p-value
    sorted_results = sorted(ticker_results, key=lambda x: x.get('p_value', 1))

    for r in sorted_results:
        ticker = r.get('ticker', '?')
        p_val = r.get('p_value', 1)
        stats = r.get('stats', {})
        perf = stats.get('performance', {})
        risk = stats.get('risk', {})
        rr = stats.get('risk_return', {})

        status = '+' if p_val < 0.05 else '-'

        lines.append(
            f"| {ticker} | {_format_value(p_val)} | "
            f"{_format_value(stats.get('trade_stats', {}).get('profit_factor', 0))} | "
            f"{_format_value(perf.get('cagr_pct', 0), 2)} | "
            f"{_format_value(risk.get('max_dd_pct', 0), 2)} | "
            f"{_format_value(rr.get('sharpe_ratio', 0))} | {status} |"
        )

    lines.extend([
        "",
        "## Aggregate Statistics",
        "",
    ])

    # Key aggregates
    key_metrics = [
        ('robustness.p_value', 'P-Value'),
        ('risk_return.sharpe_ratio', 'Sharpe'),
        ('risk_return.sqn', 'SQN'),
        ('performance.cagr_pct', 'CAGR %'),
        ('risk.max_dd_pct', 'Max DD %'),
    ]

    lines.extend([
        "| Metric | Mean | Median | Std | Min | Max |",
        "|--------|------|--------|-----|-----|-----|",
    ])

    for key, label in key_metrics:
        if key in batch_stats:
            s = batch_stats[key]
            lines.append(
                f"| {label} | {_format_value(s['mean'])} | {_format_value(s['median'])} | "
                f"{_format_value(s['std'])} | {_format_value(s['min'])} | {_format_value(s['max'])} |"
            )

    lines.extend([
        "",
        "---",
        "",
        f"*Report generated by Trading-System MCPT*",
    ])

    report_file.write_text('\n'.join(lines))
    return report_file
