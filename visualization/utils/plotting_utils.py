"""
Shared utilities for plotting and visualization.

Provides common functions used by both interactive and static visualization modules.
"""

from pathlib import Path
from typing import Optional, List, Tuple, Union
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.paths import BACKTEST_FIGURES, ensure_directories


def set_plot_style(style: str = 'seaborn-v0_8-darkgrid') -> None:
    """
    Set the matplotlib plot style.

    Args:
        style: Matplotlib style name
    """
    try:
        import matplotlib.pyplot as plt
        plt.style.use(style)
    except Exception:
        # Fallback to default style
        try:
            import matplotlib.pyplot as plt
            plt.style.use('default')
        except Exception:
            pass


def save_figure(
    fig,
    filename: str,
    dpi: int = 150,
    format: str = 'png',
    output_dir: Optional[Path] = None
) -> Path:
    """
    Save a matplotlib figure to file.

    Args:
        fig: Matplotlib figure object
        filename: Output filename (with or without extension)
        dpi: Resolution in dots per inch
        format: Output format ('png', 'pdf', 'svg', etc.)
        output_dir: Output directory (default: BACKTEST_FIGURES)

    Returns:
        Path to saved file
    """
    ensure_directories()
    output_dir = output_dir or BACKTEST_FIGURES

    # Add extension if not present
    if not filename.endswith(f'.{format}'):
        filename = f"{filename}.{format}"

    output_path = output_dir / filename
    fig.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')

    return output_path


def create_color_palette(n_colors: int = 10) -> List[str]:
    """
    Create a color palette for consistent styling.

    Args:
        n_colors: Number of colors to generate

    Returns:
        List of hex color codes
    """
    # Professional trading color palette
    base_colors = [
        '#2E86AB',  # Blue
        '#A23B72',  # Magenta
        '#F18F01',  # Orange
        '#C73E1D',  # Red
        '#3B1F2B',  # Dark purple
        '#219653',  # Green
        '#9B51E0',  # Purple
        '#2D9CDB',  # Light blue
        '#F2994A',  # Light orange
        '#EB5757',  # Light red
    ]

    if n_colors <= len(base_colors):
        return base_colors[:n_colors]

    # If more colors needed, interpolate
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        cmap = plt.cm.get_cmap('tab20', n_colors)
        return [mcolors.rgb2hex(cmap(i)) for i in range(n_colors)]
    except Exception:
        # Fallback: cycle through base colors
        return [base_colors[i % len(base_colors)] for i in range(n_colors)]


def format_currency(value: float, currency: str = 'USD', decimals: int = 2) -> str:
    """
    Format a value as currency.

    Args:
        value: Numeric value
        currency: Currency code
        decimals: Number of decimal places

    Returns:
        Formatted currency string
    """
    symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'BTC': '₿',
        'ETH': 'Ξ'
    }
    symbol = symbols.get(currency, currency)

    if abs(value) >= 1_000_000:
        return f"{symbol}{value/1_000_000:,.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{symbol}{value/1_000:,.{decimals}f}K"
    else:
        return f"{symbol}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a value as percentage.

    Args:
        value: Numeric value (0.1 = 10%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def create_subplot_layout(
    n_plots: int,
    max_cols: int = 3
) -> Tuple[int, int]:
    """
    Calculate optimal subplot layout.

    Args:
        n_plots: Number of plots
        max_cols: Maximum number of columns

    Returns:
        Tuple of (rows, cols)
    """
    import math
    cols = min(n_plots, max_cols)
    rows = math.ceil(n_plots / cols)
    return rows, cols


def add_watermark(
    fig,
    text: str = "Trading-System",
    alpha: float = 0.1
) -> None:
    """
    Add a watermark to a matplotlib figure.

    Args:
        fig: Matplotlib figure
        text: Watermark text
        alpha: Transparency (0-1)
    """
    try:
        fig.text(
            0.5, 0.5, text,
            fontsize=40, color='gray',
            ha='center', va='center',
            alpha=alpha, rotation=30,
            transform=fig.transFigure
        )
    except Exception:
        pass  # Silently fail if can't add watermark


def get_figure_size(aspect_ratio: float = 1.5, base_width: float = 10) -> Tuple[float, float]:
    """
    Calculate figure size maintaining aspect ratio.

    Args:
        aspect_ratio: Width / Height ratio
        base_width: Base width in inches

    Returns:
        Tuple of (width, height)
    """
    return (base_width, base_width / aspect_ratio)
