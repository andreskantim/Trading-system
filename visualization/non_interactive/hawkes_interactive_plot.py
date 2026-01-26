"""
Gráfico interactivo HTML con Plotly para la estrategia Hawkes

Genera un gráfico HTML interactivo con:
- Precio de Bitcoin
- Proceso de Hawkes
- Percentiles (q05, q95)
- Señales de trading
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Añadir path del proyecto
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from models.strategies import hawkes
from config.paths import ensure_ticker_output_dirs


def create_interactive_plot(ohlc: pd.DataFrame, kappa: float, lookback: int, output_path: Path):
    """
    Crea un gráfico interactivo HTML con Plotly
    """
    # Calcular componentes de la estrategia
    high = np.log(ohlc["high"])
    low = np.log(ohlc["low"])
    hl_range = high - low
    atr = hl_range.rolling(336).mean()
    norm_range = hl_range / atr
    v_hawk = hawkes.hawkes_process(norm_range, kappa)

    # Calcular percentiles
    q05 = v_hawk.rolling(lookback).quantile(0.05)
    q95 = v_hawk.rolling(lookback).quantile(0.95)

    # Generar señales
    signals = hawkes.signal(ohlc, kappa, lookback)

    # Encontrar puntos de entrada y salida
    entries_long = signals[(signals == 1) & (signals.shift(1) != 1)]
    entries_short = signals[(signals == -1) & (signals.shift(1) != -1)]
    exits = signals[(signals == 0) & (signals.shift(1) != 0)]

    # Crear subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=(
            f'Bitcoin Price & Trading Signals (kappa={kappa:.4f}, lookback={lookback}h)',
            'Hawkes Volatility Process & Percentile Thresholds',
            'Trading Signals Timeline'
        )
    )

    # 1. PRECIO DE BITCOIN
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=ohlc.index,
            open=ohlc['open'],
            high=ohlc['high'],
            low=ohlc['low'],
            close=ohlc['close'],
            name='BTC Price',
            increasing_line_color='green',
            decreasing_line_color='red',
            opacity=0.8
        ),
        row=1, col=1
    )

    # Marcadores de entrada LONG
    fig.add_trace(
        go.Scatter(
            x=entries_long.index,
            y=ohlc.loc[entries_long.index, 'close'],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='lime',
                line=dict(color='darkgreen', width=1)
            ),
            name='Long Entry',
            hovertemplate='<b>LONG ENTRY</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Marcadores de entrada SHORT
    fig.add_trace(
        go.Scatter(
            x=entries_short.index,
            y=ohlc.loc[entries_short.index, 'close'],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
                line=dict(color='darkred', width=1)
            ),
            name='Short Entry',
            hovertemplate='<b>SHORT ENTRY</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Marcadores de EXIT
    fig.add_trace(
        go.Scatter(
            x=exits.index,
            y=ohlc.loc[exits.index, 'close'],
            mode='markers',
            marker=dict(
                symbol='x',
                size=8,
                color='blue',
                line=dict(width=2)
            ),
            name='Exit',
            hovertemplate='<b>EXIT</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. PROCESO DE HAWKES Y PERCENTILES
    # Proceso de Hawkes
    fig.add_trace(
        go.Scatter(
            x=v_hawk.index,
            y=v_hawk,
            mode='lines',
            line=dict(color='purple', width=2),
            name='Hawkes Process',
            hovertemplate='Hawkes: %{y:.4f}<br>Date: %{x}<extra></extra>'
        ),
        row=2, col=1
    )

    # Percentil 5%
    fig.add_trace(
        go.Scatter(
            x=q05.index,
            y=q05,
            mode='lines',
            line=dict(color='blue', width=1, dash='dash'),
            name='5th Percentile (q05)',
            hovertemplate='q05: %{y:.4f}<br>Date: %{x}<extra></extra>'
        ),
        row=2, col=1
    )

    # Percentil 95%
    fig.add_trace(
        go.Scatter(
            x=q95.index,
            y=q95,
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            name='95th Percentile (q95)',
            hovertemplate='q95: %{y:.4f}<br>Date: %{x}<extra></extra>'
        ),
        row=2, col=1
    )

    # Área sombreada entre percentiles
    fig.add_trace(
        go.Scatter(
            x=q95.index.tolist() + q05.index.tolist()[::-1],
            y=q95.tolist() + q05.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.2)',
            line=dict(width=0),
            name='Normal Range',
            showlegend=True,
            hoverinfo='skip'
        ),
        row=2, col=1
    )

    # 3. SEÑALES DE TRADING
    # Crear áreas para long y short
    signals_df = pd.DataFrame({
        'date': signals.index,
        'signal': signals.values
    })

    # Long positions
    long_mask = signals_df['signal'] > 0
    if long_mask.any():
        fig.add_trace(
            go.Scatter(
                x=signals_df.loc[long_mask, 'date'],
                y=signals_df.loc[long_mask, 'signal'],
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.3)',
                line=dict(color='green', width=0),
                name='Long Position',
                hovertemplate='LONG<br>Date: %{x}<extra></extra>'
            ),
            row=3, col=1
        )

    # Short positions
    short_mask = signals_df['signal'] < 0
    if short_mask.any():
        fig.add_trace(
            go.Scatter(
                x=signals_df.loc[short_mask, 'date'],
                y=signals_df.loc[short_mask, 'signal'],
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red', width=0),
                name='Short Position',
                hovertemplate='SHORT<br>Date: %{x}<extra></extra>'
            ),
            row=3, col=1
        )

    # Línea en cero
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=3, col=1)

    # Actualizar ejes y layout
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price (USD)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Hawkes Value", row=2, col=1)
    fig.update_yaxes(title_text="Signal", row=3, col=1)

    # Layout general
    fig.update_layout(
        height=1200,
        showlegend=True,
        hovermode='x unified',
        title={
            'text': f'<b>Hawkes Volatility Strategy - Interactive Analysis</b><br>' +
                   f'<sub>Parameters: kappa={kappa:.4f}, lookback={lookback} hours | Period: {ohlc.index[0].date()} to {ohlc.index[-1].date()}</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        )
    )

    # Configurar rangeslider para el precio
    fig.update_xaxes(rangeslider_visible=False)

    # Guardar como HTML
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  HTML: {output_path}")

    # También guardar una versión estática PNG
    try:
        import plotly.io as pio
        png_path = output_path.with_suffix('.png')
        pio.write_image(fig, str(png_path), width=1920, height=1200)
        print(f"  PNG: {png_path}")
    except Exception as e:
        print(f"  PNG export failed (requires kaleido): {e}")


def main():
    """Función principal"""
    print("\n" + "="*70)
    print("GRÁFICO INTERACTIVO HAWKES STRATEGY")
    print("="*70)

    ticker = 'BTC'
    strategy = 'hawkes'

    # Leer mejores parámetros del análisis anterior
    output_dirs = ensure_ticker_output_dirs(strategy, ticker)
    best_params_file = output_dirs['results'] / "best_parameters.txt"

    if best_params_file.exists():
        print(f"\nLeyendo mejores parámetros desde {best_params_file}...")
        with open(best_params_file, 'r') as f:
            lines = f.readlines()
            kappa = float(lines[6].split(':')[1].strip())
            lookback = int(float(lines[7].split(':')[1].split()[0].strip()))
        print(f"Parámetros cargados: kappa={kappa:.4f}, lookback={lookback}")
    else:
        kappa = 0.125
        lookback = 120
        print(f"Usando parámetros por defecto: kappa={kappa}, lookback={lookback}")

    # Cargar datos
    data_path = project_root / "data" / "operative" / f"{ticker}.parquet"
    if not data_path.exists():
        data_path = project_root / "data" / "BTCUSD" / "BTCUSD3600.pq"
    print(f"\nCargando datos desde {data_path}...")
    df = pd.read_parquet(data_path)

    # Período in-sample: 2018-2022
    insample_start = "2018-01-01"
    insample_end = "2022-12-31"
    df_insample = df.loc[insample_start:insample_end]
    print(f"Datos cargados: {len(df_insample)} barras")

    # Generar gráfico interactivo - save to figures/
    output_file = output_dirs['figures'] / f"{ticker}_hawkes_interactive.html"
    print(f"\nGenerando gráfico interactivo...")
    create_interactive_plot(df_insample, kappa, lookback, output_file)

    print("\n" + "="*70)
    print("GRÁFICO INTERACTIVO COMPLETADO")
    print("="*70)
    print(f"\nArchivo generado: {output_file}")
    print()


if __name__ == "__main__":
    main()
