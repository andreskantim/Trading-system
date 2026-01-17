"""
Lightweight Charts viewer for interactive trading data visualization.

Provides candlestick charts with trading signals and indicators
using the lightweight-charts library via Python bindings.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.paths import (
    DATA_DIR,
    BACKTEST_FIGURES,
    ensure_directories
)


class LightweightChartsViewer:
    """
    Interactive chart viewer using lightweight-charts.

    Provides methods for:
    - Candlestick charts with OHLC data
    - Volume bars
    - Trading signals overlay
    - Technical indicators
    """

    def __init__(self, width: int = 1200, height: int = 600):
        """
        Initialize the viewer.

        Args:
            width: Chart width in pixels
            height: Chart height in pixels
        """
        self.width = width
        self.height = height
        self._chart = None
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        try:
            from lightweight_charts import Chart
            self._has_lightweight_charts = True
        except ImportError:
            self._has_lightweight_charts = False
            print("Warning: lightweight-charts not installed. "
                  "Install with: pip install lightweight-charts")

    def create_candlestick_chart(
        self,
        data: pd.DataFrame,
        title: str = "Trading Chart"
    ):
        """
        Create an interactive candlestick chart.

        Args:
            data: DataFrame with columns ['time', 'open', 'high', 'low', 'close', 'volume']
            title: Chart title

        Returns:
            Chart object if available, None otherwise
        """
        if not self._has_lightweight_charts:
            print("Cannot create chart: lightweight-charts not installed")
            return None

        from lightweight_charts import Chart

        chart = Chart(width=self.width, height=self.height)
        chart.set(data)

        self._chart = chart
        return chart

    def add_signals(
        self,
        signals: pd.DataFrame,
        buy_color: str = 'green',
        sell_color: str = 'red'
    ) -> None:
        """
        Add buy/sell signals to the chart.

        Args:
            signals: DataFrame with columns ['time', 'signal']
                     where signal is 1 for buy, -1 for sell
            buy_color: Color for buy signals
            sell_color: Color for sell signals
        """
        if self._chart is None:
            print("No chart created yet. Call create_candlestick_chart first.")
            return

        for _, row in signals.iterrows():
            if row['signal'] == 1:
                self._chart.marker(time=row['time'], color=buy_color, shape='arrowUp')
            elif row['signal'] == -1:
                self._chart.marker(time=row['time'], color=sell_color, shape='arrowDown')

    def add_line(
        self,
        data: pd.DataFrame,
        name: str = "indicator",
        color: str = 'blue',
        width: int = 2
    ) -> None:
        """
        Add a line indicator to the chart.

        Args:
            data: DataFrame with columns ['time', 'value']
            name: Name of the indicator
            color: Line color
            width: Line width
        """
        if self._chart is None:
            print("No chart created yet. Call create_candlestick_chart first.")
            return

        line = self._chart.create_line(name=name, color=color, width=width)
        line.set(data)

    def show(self, block: bool = True) -> None:
        """
        Display the chart.

        Args:
            block: Whether to block execution until chart is closed
        """
        if self._chart is None:
            print("No chart created yet.")
            return

        self._chart.show(block=block)

    def save_screenshot(self, filename: str) -> Optional[Path]:
        """
        Save a screenshot of the chart.

        Args:
            filename: Output filename

        Returns:
            Path to saved file, or None if failed
        """
        if self._chart is None:
            print("No chart created yet.")
            return None

        ensure_directories()
        output_path = BACKTEST_FIGURES / filename
        self._chart.screenshot(str(output_path))
        return output_path


def load_ohlc_data(symbol: str = "BTCUSD") -> pd.DataFrame:
    """
    Load OHLC data for a given symbol.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSD')

    Returns:
        DataFrame with OHLC data
    """
    from config.paths import BITCOIN_PARQUET, BITCOIN_CSV

    # Try parquet first
    if BITCOIN_PARQUET.exists():
        df = pd.read_parquet(BITCOIN_PARQUET)
    elif BITCOIN_CSV.exists():
        df = pd.read_csv(BITCOIN_CSV)
    else:
        raise FileNotFoundError(f"No data file found for {symbol}")

    # Ensure proper column names
    column_mapping = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    df = df.rename(columns=column_mapping)

    # Ensure time column
    if 'time' not in df.columns:
        if 'timestamp' in df.columns:
            df['time'] = pd.to_datetime(df['timestamp'])
        elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            df['time'] = df.index
        else:
            df['time'] = df.index

    return df


def create_interactive_chart(
    ohlc_data: pd.DataFrame,
    vis_data: Dict[str, Any],
    strategy_name: str,
    params: tuple,
    output_path: Path
) -> Optional[Path]:
    """
    Generic function to create interactive charts for any strategy.

    Args:
        ohlc_data: DataFrame with OHLC data (index should be DatetimeIndex)
        vis_data: Dict returned by strategy.visualization() with keys:
            - 'indicators': dict mapping name -> {'data': Series, 'color': str, 'panel': str}
            - 'signals': Series with 1 (long), -1 (short), 0 (flat)
        strategy_name: Name of strategy (for title)
        params: Tuple of optimized parameters
        output_path: Where to save HTML file

    Returns:
        Path to saved file, or None if failed
    """
    try:
        from lightweight_charts import Chart
    except ImportError:
        print("Warning: lightweight-charts not installed.")
        print("Install with: pip install lightweight-charts")
        return None

    # Prepare OHLC data for lightweight-charts
    df = ohlc_data.copy()

    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert index to 'time' column if needed
    if 'time' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'time'})
        else:
            df['time'] = df.index

    # Create the main chart
    chart = Chart(width=1400, height=800)

    # Set OHLC data
    chart.set(df[['time', 'open', 'high', 'low', 'close']])

    # Process indicators
    indicators = vis_data.get('indicators', {})
    lower_panel_lines = []

    for name, ind_spec in indicators.items():
        data = ind_spec.get('data')
        color = ind_spec.get('color', 'blue')
        panel = ind_spec.get('panel', 'price')

        if data is None or data.empty:
            continue

        # Prepare indicator data
        ind_df = pd.DataFrame({
            'time': data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data.index),
            'value': data.values
        }).dropna()

        if panel == 'price':
            # Overlay on price chart
            line = chart.create_line(
                name=name.replace('_', ' ').title(),
                color=color,
                width=2
            )
            line.set(ind_df)
        elif panel == 'lower':
            # Will be added to subchart
            lower_panel_lines.append({
                'name': name,
                'data': ind_df,
                'color': color
            })

    # Create subchart for lower panel indicators
    if lower_panel_lines:
        subchart = chart.create_subchart(
            width=1,
            height=0.3,
            sync=True
        )
        for line_spec in lower_panel_lines:
            line = subchart.create_line(
                name=line_spec['name'].replace('_', ' ').title(),
                color=line_spec['color'],
                width=2
            )
            line.set(line_spec['data'])

    # Add signal markers
    signals = vis_data.get('signals', pd.Series())
    if not signals.empty:
        # Detect signal changes (entries/exits)
        signal_changes = signals.diff().fillna(0)

        for idx in signals.index:
            sig_val = signals.loc[idx]
            change = signal_changes.loc[idx]

            # Long entry (signal goes from non-1 to 1)
            if sig_val == 1 and change != 0:
                try:
                    chart.marker(
                        time=idx,
                        position='below',
                        shape='arrow_up',
                        color='green',
                        text='LONG'
                    )
                except Exception:
                    pass  # Skip if marker fails

            # Short entry (signal goes from non-(-1) to -1)
            elif sig_val == -1 and change != 0:
                try:
                    chart.marker(
                        time=idx,
                        position='above',
                        shape='arrow_down',
                        color='red',
                        text='SHORT'
                    )
                except Exception:
                    pass

            # Exit (signal goes to 0 from non-zero)
            elif sig_val == 0 and change != 0:
                try:
                    chart.marker(
                        time=idx,
                        position='inside',
                        shape='circle',
                        color='yellow',
                        text='EXIT'
                    )
                except Exception:
                    pass

    # Add watermark with strategy name and parameters
    param_str = ', '.join([
        f'{p:.3f}' if isinstance(p, float) else str(p)
        for p in params
    ])
    chart.watermark(f'{strategy_name.upper()} | Params: {param_str}')

    # Save to HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        chart.save(str(output_path))
        return output_path
    except Exception as e:
        print(f"Error saving chart: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    print("Lightweight Charts Viewer")
    print("=" * 50)

    viewer = LightweightChartsViewer()

    try:
        data = load_ohlc_data("BTCUSD")
        print(f"Loaded {len(data)} rows of data")

        if viewer._has_lightweight_charts:
            chart = viewer.create_candlestick_chart(data, title="BTCUSD Hourly")
            viewer.show()
        else:
            print("Install lightweight-charts to view interactive charts:")
            print("  pip install lightweight-charts")
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
