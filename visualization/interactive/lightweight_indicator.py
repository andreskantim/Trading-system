"""
Lightweight Charts Viewer para Indicadores Técnicos

Genera visualización HTML con TradingView Lightweight Charts mostrando
OHLC + múltiples indicadores técnicos (SIN señales de trading).

Diferencias con lightweight_strategy.py:
- NO genera trade markers
- NO genera trade shading overlays
- NO procesa señales de trading
- Enfocado solo en visualización de indicadores
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import sys
import json
import http.server
import socketserver
import subprocess
import socket

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.paths import ensure_directories


def find_free_port(start_port=8000, max_attempts=10):
    """Encuentra un puerto libre."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No puerto libre encontrado entre {start_port}-{start_port+max_attempts}")


def serve_and_open(filepath: Path):
    """Sirve HTML y abre navegador (WSL compatible)."""

    port = find_free_port()

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(filepath.parent), **kwargs)

        def log_message(self, format, *args):
            pass

    httpd = socketserver.TCPServer(("", port), QuietHandler)
    httpd.allow_reuse_address = True

    url = f"http://localhost:{port}/{filepath.name}"

    print(f"{'='*60}")
    print(f"Servidor: http://localhost:{port}")
    print(f"{'='*60}\n")

    # Abrir navegador (WSL -> Windows)
    try:
        subprocess.Popen(['cmd.exe', '/c', 'start', url],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        print(f"Navegador abierto automaticamente")
    except Exception:
        print(f"Abre manualmente: {url}")

    print(f"\nPresiona Ctrl+C para cerrar\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nCerrando...")
        httpd.shutdown()
        httpd.server_close()
        print("Cerrado\n")


def create_indicator_chart(
    ohlc_data: pd.DataFrame,
    vis_data: Dict[str, Any],
    indicator_names: List[str],
    output_path: Path = None
) -> Optional[Path]:
    """
    Genera gráfico HTML interactivo SOLO con indicadores (sin trading signals).

    Args:
        ohlc_data: DataFrame con OHLC data (index debe ser DatetimeIndex)
        vis_data: Dict con 'indicators_in_price', 'indicators_off_price'
        indicator_names: Lista de nombres de indicadores para título
        output_path: Ruta donde guardar HTML (opcional)

    Returns:
        Path al archivo HTML generado
    """

    df = ohlc_data.copy()

    # Validar columnas requeridas
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must have columns: {required_cols}")

    # Asegurar que el índice es DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index)

    # Extraer componentes
    indicators_in_price = vis_data.get('indicators_in_price', {})
    indicators_off_price = vis_data.get('indicators_off_price', {})

    # Título: "RSI + Stochastic + Moving Average"
    indicator_title = " + ".join([ind.replace('_', ' ').title() for ind in indicator_names])

    print(f"\n{'='*60}")
    print(f"INDICADORES: {indicator_title}")
    print(f"Datos: {len(df):,} barras")
    print(f"{'='*60}\n")

    # ==========================================
    # 1. Preparar datos OHLC con timestamp Unix
    # ==========================================
    df_ohlc = df.copy()
    df_ohlc['time'] = (df_ohlc.index.astype(int) // 10**9).astype(int)
    df_ohlc = df_ohlc.dropna(subset=['open', 'high', 'low', 'close'])
    df_ohlc = df_ohlc.sort_values('time')
    df_ohlc = df_ohlc.drop_duplicates(subset=['time'], keep='first')

    ohlc_json = []
    for _, row in df_ohlc[['time', 'open', 'high', 'low', 'close']].iterrows():
        ohlc_json.append({
            'time': int(row['time']),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        })

    print(f"OHLC: {len(ohlc_json):,} puntos")

    # ==========================================
    # 2. Preparar indicadores in-price
    # ==========================================
    indicators_in_json = []
    for name, spec in indicators_in_price.items():
        data = spec.get('data')
        if data is None or data.empty:
            continue

        display_name = name.replace('_', ' ').title()

        # Eliminar NaN directamente de la Serie
        data_clean = data.dropna()

        if len(data_clean) == 0:
            print(f"{display_name}: todos NaN, omitiendo")
            continue

        # Convertir a timestamp Unix (ahora sin NaN)
        ind_df = pd.DataFrame({
            'time': (pd.to_datetime(data_clean.index).astype(int) // 10**9).astype(int),
            'value': data_clean.values
        })
        ind_df = ind_df.drop_duplicates(subset=['time'], keep='first')
        ind_df = ind_df.sort_values('time')

        indicators_in_json.append({
            'name': display_name,
            'color': spec.get('color', 'cyan'),
            'data': ind_df.to_dict('records')
        })

        # Debug: mostrar rango temporal del indicador
        first_date = data_clean.index[0]
        last_date = data_clean.index[-1]
        print(f"{display_name}: {len(ind_df)} puntos ({first_date} a {last_date})")

    # ==========================================
    # 3. Preparar indicadores off-price
    # ==========================================
    indicators_off_json = []
    for name, spec in indicators_off_price.items():
        data = spec.get('data')
        if data is None or data.empty:
            continue

        display_name = name.replace('_', ' ').title()

        # Reindexar al índice completo de OHLC para tener los mismos timestamps
        data_reindexed = data.reindex(df.index)

        # Convertir directamente a timestamp Unix
        ind_df = pd.DataFrame({
            'time': (pd.to_datetime(data_reindexed.index).astype(int) // 10**9).astype(int),
            'value': data_reindexed.values
        })
        ind_df = ind_df.drop_duplicates(subset=['time'], keep='first')
        ind_df = ind_df.sort_values('time')

        # Reemplazar NaN por None para JSON (se convierte a null)
        ind_df['value'] = ind_df['value'].replace({np.nan: None})

        indicators_off_json.append({
            'name': display_name,
            'color': spec.get('color', 'orange'),
            'data': ind_df.to_dict('records')
        })

        # Debug
        valid_count = data.notna().sum()
        first_valid_idx = data.first_valid_index()
        last_valid_idx = data.last_valid_index()
        print(f"{display_name} (subchart): {len(ind_df)} puntos totales, {valid_count} validos")
        if first_valid_idx and last_valid_idx:
            print(f"    Rango valido: {first_valid_idx} a {last_valid_idx}")

    # ==========================================
    # 4. Guardar archivos JSON
    # ==========================================
    if output_path is None:
        ensure_directories()
        output_dir = BACKTEST_FIGURES / 'indicators'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'indicator_chart.html'

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar JSON en archivos separados
    data_file = output_path.parent / 'chart_data.json'
    indicators_in_file = output_path.parent / 'indicators_in.json'
    indicators_off_file = output_path.parent / 'indicators_off.json'

    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(ohlc_json, f, ensure_ascii=False, allow_nan=False)

    with open(indicators_in_file, 'w', encoding='utf-8') as f:
        json.dump(indicators_in_json, f, ensure_ascii=False, allow_nan=False)

    with open(indicators_off_file, 'w', encoding='utf-8') as f:
        json.dump(indicators_off_json, f, ensure_ascii=False, allow_nan=False)

    has_subchart = len(indicators_off_json) > 0

    # ==========================================
    # 5. Generar HTML
    # ==========================================
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{indicator_title}</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            overflow: hidden;
        }}
        #container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
            width: 100vw;
        }}
        #main-chart {{ flex: {"7" if has_subchart else "1"}; }}
        #sub-chart {{ flex: 3; border-top: 1px solid #30363d; }}
        .info {{
            position: absolute;
            top: 12px;
            left: 12px;
            background: rgba(13, 17, 23, 0.95);
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 14px;
            z-index: 1000;
            border: 1px solid #30363d;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }}
    </style>
</head>
<body>
    <div class="info"><strong>{indicator_title}</strong></div>
    <div id="container">
        <div id="main-chart"></div>
        {"<div id='sub-chart'></div>" if has_subchart else ""}
    </div>
    <script>
        // Cargar todos los datos desde archivos JSON
        Promise.all([
            fetch('chart_data.json').then(r => r.json()),
            fetch('indicators_in.json').then(r => r.json()),
            fetch('indicators_off.json').then(r => r.json())
        ]).then(([ohlcData, indicatorsIn, indicatorsOff]) => {{
            console.log('OHLC:', ohlcData.length, 'points');
            console.log('In-price indicators:', indicatorsIn.length);
            console.log('Off-price indicators:', indicatorsOff.length);

            // Main chart
            const mainChart = LightweightCharts.createChart(document.getElementById('main-chart'), {{
                width: window.innerWidth,
                height: {"window.innerHeight * 0.7" if has_subchart else "window.innerHeight"},
                layout: {{
                    background: {{ color: '#0d1117' }},
                    textColor: '#c9d1d9',
                    fontSize: 12,
                }},
                grid: {{
                    vertLines: {{ color: 'rgba(197, 203, 206, 0.1)' }},
                    horzLines: {{ color: 'rgba(197, 203, 206, 0.1)' }},
                }},
                crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                timeScale: {{
                    timeVisible: true,
                    secondsVisible: false,
                    minBarSpacing: 0.001,
                    rightOffset: 5,
                    barSpacing: 6,
                }},
                rightPriceScale: {{
                    borderVisible: false,
                    scaleMargins: {{
                        top: 0.1,
                        bottom: 0.1,
                    }},
                    minimumWidth: 80,
                    alignLabels: true,
                }},
            }});

            // Candlestick series
            const candleSeries = mainChart.addCandlestickSeries({{
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderUpColor: '#26a69a',
                borderDownColor: '#ef5350',
                wickUpColor: '#26a69a',
                wickDownColor: '#ef5350',
                priceLineVisible: false,
            }});
            candleSeries.setData(ohlcData);

            // In-price indicators
            indicatorsIn.forEach(ind => {{
                const line = mainChart.addLineSeries({{
                    color: ind.color,
                    lineWidth: 2,
                    title: ind.name,
                }});
                line.setData(ind.data);
            }});

            {"" if not has_subchart else '''
            // Subchart
            const subChart = LightweightCharts.createChart(document.getElementById('sub-chart'), {
                width: window.innerWidth,
                height: window.innerHeight * 0.3,
                layout: {
                    background: { color: '#0d1117' },
                    textColor: '#c9d1d9',
                    fontSize: 12,
                },
                grid: {
                    vertLines: { color: 'rgba(197, 203, 206, 0.1)' },
                    horzLines: { color: 'rgba(197, 203, 206, 0.1)' },
                },
                timeScale: {
                    timeVisible: true,
                    secondsVisible: false,
                    minBarSpacing: 0.001,
                    rightOffset: 5,
                    barSpacing: 6,
                },
                rightPriceScale: {
                    borderVisible: false,
                    scaleMargins: {
                        top: 0.1,
                        bottom: 0.1,
                    },
                    minimumWidth: 80,
                    alignLabels: true,
                },
            });

            // Off-price indicators
            indicatorsOff.forEach(ind => {
                const line = subChart.addLineSeries({
                    color: ind.color,
                    lineWidth: 2,
                    title: ind.name,
                    priceLineVisible: false,
                    lastValueVisible: true,
                });

                // Asegurar que el primer timestamp coincida con OHLC
                const paddedData = [...ind.data];

                if (paddedData.length > 0 && ohlcData.length > 0) {
                    const firstOhlcTime = ohlcData[0].time;
                    const firstIndTime = paddedData[0].time;

                    if (firstIndTime > firstOhlcTime) {
                        paddedData.unshift({ time: firstOhlcTime, value: null });
                    }
                }

                line.setData(paddedData);
            });

            // Sync timeframes
            mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
                if (range) subChart.timeScale().setVisibleLogicalRange(range);
            });
            subChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
                if (range) mainChart.timeScale().setVisibleLogicalRange(range);
            });
            '''}


            // Responsive
            window.addEventListener('resize', () => {{
                mainChart.applyOptions({{
                    width: window.innerWidth,
                    height: {"window.innerHeight * 0.7" if has_subchart else "window.innerHeight"}
                }});
                {"subChart.applyOptions({ width: window.innerWidth, height: window.innerHeight * 0.3 });" if has_subchart else ""}
            }});

            console.log('Chart loaded successfully!');
        }}).catch(error => {{
            console.error('Error loading data:', error);
        }});
    </script>
</body>
</html>'''

    # Guardar HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nGuardado: {output_path}\n")

    # Servir y abrir
    serve_and_open(output_path)

    return output_path


if __name__ == "__main__":
    print("Lightweight Charts Indicator Viewer")
