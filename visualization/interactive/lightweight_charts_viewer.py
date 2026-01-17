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
        title: str = "Trading Chart",
        show_volume: bool = True
    ):
        """
        Create an interactive candlestick chart.

        Args:
            data: DataFrame with columns ['time', 'open', 'high', 'low', 'close', 'volume']
            title: Chart title
            show_volume: Whether to show volume bars

        Returns:
            Chart object if available, None otherwise
        """
        if not self._has_lightweight_charts:
            print("Cannot create chart: lightweight-charts not installed")
            return None

        from lightweight_charts import Chart

        chart = Chart(width=self.width, height=self.height)
        chart.set(data)

        if show_volume and 'volume' in data.columns:
            chart.volume_config(enabled=True)

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
