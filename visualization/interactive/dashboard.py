"""
Dashboard for MCPT results visualization.

Provides an interactive dashboard for viewing backtest results,
Monte Carlo permutation test outcomes, and walk-forward analysis.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.paths import (
    BACKTEST_RESULTS,
    BACKTEST_FIGURES,
    BACKTEST_REPORTS,
    ensure_directories
)


class MCPTDashboard:
    """
    Dashboard for visualizing MCPT (Monte Carlo Permutation Test) results.

    Features:
    - P-value distribution charts
    - Return comparison plots
    - Walk-forward analysis visualization
    - Statistical summary tables
    """

    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize the dashboard.

        Args:
            results_dir: Directory containing MCPT results (default: BACKTEST_RESULTS)
        """
        self.results_dir = results_dir or BACKTEST_RESULTS
        self._results: Dict[str, pd.DataFrame] = {}
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        self._has_plotly = False
        self._has_streamlit = False
        self._has_dash = False

        try:
            import plotly.graph_objects as go
            import plotly.express as px
            self._has_plotly = True
        except ImportError:
            print("Warning: plotly not installed. Install with: pip install plotly")

        try:
            import streamlit
            self._has_streamlit = True
        except ImportError:
            pass  # Streamlit is optional

        try:
            import dash
            self._has_dash = True
        except ImportError:
            pass  # Dash is optional

    def load_results(self, filename: str = "mcpt_results.csv") -> pd.DataFrame:
        """
        Load MCPT results from file.

        Args:
            filename: Name of results file

        Returns:
            DataFrame with results
        """
        filepath = self.results_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        df = pd.read_csv(filepath)
        self._results[filename] = df
        return df

    def plot_pvalue_distribution(
        self,
        results: pd.DataFrame,
        title: str = "P-Value Distribution"
    ):
        """
        Create a histogram of p-values.

        Args:
            results: DataFrame with 'pvalue' column
            title: Chart title

        Returns:
            Plotly figure object
        """
        if not self._has_plotly:
            print("Plotly not installed. Cannot create plot.")
            return None

        import plotly.express as px

        fig = px.histogram(
            results,
            x='pvalue',
            nbins=50,
            title=title,
            labels={'pvalue': 'P-Value', 'count': 'Frequency'}
        )

        # Add significance threshold line
        fig.add_vline(x=0.05, line_dash="dash", line_color="red",
                      annotation_text="α = 0.05")

        return fig

    def plot_returns_comparison(
        self,
        results: pd.DataFrame,
        title: str = "Strategy vs Permuted Returns"
    ):
        """
        Create a comparison plot of strategy returns vs permuted returns.

        Args:
            results: DataFrame with 'strategy_return' and 'permuted_returns' columns
            title: Chart title

        Returns:
            Plotly figure object
        """
        if not self._has_plotly:
            print("Plotly not installed. Cannot create plot.")
            return None

        import plotly.graph_objects as go

        fig = go.Figure()

        # Add permuted returns distribution
        if 'permuted_returns' in results.columns:
            fig.add_trace(go.Histogram(
                x=results['permuted_returns'],
                name='Permuted Returns',
                opacity=0.7
            ))

        # Add strategy return line
        if 'strategy_return' in results.columns:
            strategy_return = results['strategy_return'].iloc[0]
            fig.add_vline(
                x=strategy_return,
                line_dash="solid",
                line_color="green",
                annotation_text=f"Strategy: {strategy_return:.2%}"
            )

        fig.update_layout(title=title)
        return fig

    def create_summary_table(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Create a summary table of MCPT results.

        Args:
            results: DataFrame with MCPT results

        Returns:
            Summary DataFrame
        """
        summary = {
            'Metric': [],
            'Value': []
        }

        if 'pvalue' in results.columns:
            summary['Metric'].append('Mean P-Value')
            summary['Value'].append(f"{results['pvalue'].mean():.4f}")

            summary['Metric'].append('Significant Tests (p < 0.05)')
            sig_count = (results['pvalue'] < 0.05).sum()
            summary['Value'].append(f"{sig_count} / {len(results)}")

        if 'strategy_return' in results.columns:
            summary['Metric'].append('Strategy Return')
            summary['Value'].append(f"{results['strategy_return'].mean():.2%}")

        if 'sharpe_ratio' in results.columns:
            summary['Metric'].append('Sharpe Ratio')
            summary['Value'].append(f"{results['sharpe_ratio'].mean():.2f}")

        return pd.DataFrame(summary)

    def save_report(
        self,
        results: pd.DataFrame,
        filename: str = "mcpt_report.html"
    ) -> Path:
        """
        Save an HTML report of the results.

        Args:
            results: DataFrame with results
            filename: Output filename

        Returns:
            Path to saved report
        """
        ensure_directories()
        output_path = BACKTEST_REPORTS / filename

        if not self._has_plotly:
            # Fallback to simple HTML
            html_content = f"""
            <html>
            <head><title>MCPT Report</title></head>
            <body>
            <h1>MCPT Results Report</h1>
            {self.create_summary_table(results).to_html()}
            </body>
            </html>
            """
        else:
            import plotly.io as pio

            # Create figures
            pvalue_fig = self.plot_pvalue_distribution(results)

            html_content = f"""
            <html>
            <head>
            <title>MCPT Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
            <h1>MCPT Results Report</h1>
            <h2>Summary</h2>
            {self.create_summary_table(results).to_html()}
            <h2>P-Value Distribution</h2>
            {pio.to_html(pvalue_fig, full_html=False) if pvalue_fig else ''}
            </body>
            </html>
            """

        with open(output_path, 'w') as f:
            f.write(html_content)

        return output_path


def get_available_results() -> List[Path]:
    """
    Get list of available result files.

    Returns:
        List of paths to result files
    """
    ensure_directories()
    return list(BACKTEST_RESULTS.glob("*.csv"))


if __name__ == "__main__":
    print("MCPT Dashboard")
    print("=" * 50)

    dashboard = MCPTDashboard()

    # Show available results
    results = get_available_results()
    if results:
        print(f"Found {len(results)} result files:")
        for r in results:
            print(f"  - {r.name}")
    else:
        print("No result files found in:", BACKTEST_RESULTS)
        print("\nRun backtest scripts to generate results first.")
