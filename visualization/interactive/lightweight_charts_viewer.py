"""
Lightweight Charts Viewer - Versión Final

Genera HTML con TradingView Lightweight Charts usando timestamp Unix.
Compatible con WSL, abre navegador automáticamente.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import sys
import json
import http.server
import socketserver
import subprocess
import socket

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.paths import BACKTEST_FIGURES, ensure_directories


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


def create_interactive_chart(
    ohlc_data: pd.DataFrame,
    vis_data: Dict[str, Any],
    strategy_name: str,
    params: tuple,
    output_path: Path = None
) -> Optional[Path]:
    """
    Genera gráfico HTML interactivo con Lightweight Charts.
    
    Args:
        ohlc_data: DataFrame con OHLC data (index debe ser DatetimeIndex)
        vis_data: Dict con 'indicators_in_price', 'indicators_off_price', 'signals'
        strategy_name: Nombre de la estrategia
        params: Tupla de parámetros
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
    signals = vis_data.get('signals', pd.Series())
    
    param_str = ', '.join([f'{p:.3f}' if isinstance(p, float) else str(p) for p in params])
    
    print(f"\n{'='*60}")
    print(f"📊 {strategy_name.upper()} | {param_str}")
    print(f"📈 Datos: {len(df):,} barras")
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
    
    print(f"✓ OHLC: {len(ohlc_json):,} puntos")
    
    # ==========================================
    # 2. Preparar indicadores in-price
    # ==========================================
    indicators_in_json = []
    for name, spec in indicators_in_price.items():
        data = spec.get('data')
        if data is None or data.empty:
            continue
        
        display_name = name.replace('_', ' ').title()
        
        # Convertir a timestamp Unix
        ind_df = pd.DataFrame({
            'time': (pd.to_datetime(data.index).astype(int) // 10**9).astype(int),
            'value': data.values
        }).dropna()
        ind_df = ind_df.drop_duplicates(subset=['time'], keep='first')
        ind_df = ind_df.sort_values('time')
        
        indicators_in_json.append({
            'name': display_name,
            'color': spec.get('color', 'cyan'),
            'data': ind_df.to_dict('records')
        })
        print(f"✓ {display_name}: {len(ind_df)} puntos")
    
    # ==========================================
    # 3. Preparar indicadores off-price
    # ==========================================
    indicators_off_json = []
    for name, spec in indicators_off_price.items():
        data = spec.get('data')
        if data is None or data.empty:
            continue
        
        display_name = name.replace('_', ' ').title()
        
        # Convertir a timestamp Unix
        ind_df = pd.DataFrame({
            'time': (pd.to_datetime(data.index).astype(int) // 10**9).astype(int),
            'value': data.values
        }).dropna()
        ind_df = ind_df.drop_duplicates(subset=['time'], keep='first')
        ind_df = ind_df.sort_values('time')
        
        indicators_off_json.append({
            'name': display_name,
            'color': spec.get('color', 'orange'),
            'data': ind_df.to_dict('records')
        })
        print(f"✓ {display_name} (subchart): {len(ind_df)} puntos")
    
    # ==========================================
    # 4. Preparar marcadores de señales
    # ==========================================
    markers = []
    if not signals.empty:
        signal_changes = signals.diff().fillna(0)
        for idx in signals.index:
            sig_val = signals.loc[idx]
            change = signal_changes.loc[idx]
            timestamp = int(pd.Timestamp(idx).value // 10**9)
            
            if sig_val == 1 and change != 0:
                markers.append({
                    'time': timestamp,
                    'position': 'belowBar',
                    'color': '#26a69a',
                    'shape': 'arrowUp',
                    'text': 'L'
                })
            elif sig_val == -1 and change != 0:
                markers.append({
                    'time': timestamp,
                    'position': 'aboveBar',
                    'color': '#ef5350',
                    'shape': 'arrowDown',
                    'text': 'S'
                })
        print(f"✓ {len(markers)} señales")
    
    # ==========================================
    # 5. Guardar archivos JSON
    # ==========================================
    if output_path is None:
        ensure_directories()
        output_dir = BACKTEST_FIGURES / strategy_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'interactive_chart.html'
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar JSON en archivos separados
    data_file = output_path.parent / 'chart_data.json'
    indicators_in_file = output_path.parent / 'indicators_in.json'
    indicators_off_file = output_path.parent / 'indicators_off.json'
    markers_file = output_path.parent / 'markers.json'
    
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(ohlc_json, f, ensure_ascii=False, allow_nan=False)
    
    with open(indicators_in_file, 'w', encoding='utf-8') as f:
        json.dump(indicators_in_json, f, ensure_ascii=False, allow_nan=False)
    
    with open(indicators_off_file, 'w', encoding='utf-8') as f:
        json.dump(indicators_off_json, f, ensure_ascii=False, allow_nan=False)
    
    with open(markers_file, 'w', encoding='utf-8') as f:
        json.dump(markers, f, ensure_ascii=False, allow_nan=False)
    
    # DEBUG: Verificar qué se guardó
    print(f"\n📂 Archivos JSON guardados:")
    print(f"  - chart_data.json: {len(ohlc_json)} puntos")
    print(f"  - indicators_in.json: {len(indicators_in_json)} indicadores")
    print(f"  - indicators_off.json: {len(indicators_off_json)} indicadores")
    for idx, ind in enumerate(indicators_off_json):
        print(f"      [{idx}] {ind['name']}: {len(ind['data'])} puntos")
    print(f"  - markers.json: {len(markers)} marcadores")
    
    has_subchart = len(indicators_off_json) > 0
    
    # ==========================================
    # 6. Generar HTML
    # ==========================================
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{strategy_name.upper()} - {param_str}</title>
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
    <div class="info"><strong>{strategy_name.upper()}</strong> | {param_str}</div>
    <div id="container">
        <div id="main-chart"></div>
        {"<div id='sub-chart'></div>" if has_subchart else ""}
    </div>
    <script>
        // Cargar todos los datos desde archivos JSON
        Promise.all([
            fetch('chart_data.json').then(r => r.json()),
            fetch('indicators_in.json').then(r => r.json()),
            fetch('indicators_off.json').then(r => r.json()),
            fetch('markers.json').then(r => r.json())
        ]).then(([ohlcData, indicatorsIn, indicatorsOff, markers]) => {{
            console.log('OHLC:', ohlcData.length, 'points');
            console.log('In-price indicators:', indicatorsIn.length);
            console.log('Off-price indicators:', indicatorsOff.length);
            console.log('Markers:', markers.length);
            
            // Main chart
            const mainChart = LightweightCharts.createChart(document.getElementById('main-chart'), {{
                width: window.innerWidth,
                height: {"window.innerHeight * 0.7" if has_subchart else "window.innerHeight"},
                layout: {{
                    background: {{ color: '#0d1117' }},
                    textColor: '#c9d1d9',
                }},
                grid: {{
                    vertLines: {{ color: 'rgba(197, 203, 206, 0.1)' }},
                    horzLines: {{ color: 'rgba(197, 203, 206, 0.1)' }},
                }},
                crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                timeScale: {{ timeVisible: true, secondsVisible: false }},
            }});
            
            // Candlestick series (API v4)
            const candleSeries = mainChart.addCandlestickSeries({{
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderUpColor: '#26a69a',
                borderDownColor: '#ef5350',
                wickUpColor: '#26a69a',
                wickDownColor: '#ef5350',
            }});
            candleSeries.setData(ohlcData);
            
            // Markers
            if (markers.length > 0) {{
                console.log('📍 Aplicando', markers.length, 'markers');
                console.log('Muestra:', markers.slice(0, 2));
                
                if (typeof candleSeries.setMarkers === 'function') {{
                    try {{
                        candleSeries.setMarkers(markers);
                        console.log('✓ Markers aplicados correctamente');
                    }} catch (e) {{
                        console.error('❌ Error aplicando markers:', e.message);
                    }}
                }} else {{
                    console.error('❌ setMarkers no está disponible');
                    console.log('Métodos disponibles:', Object.keys(candleSeries));
                }}
            }} else {{
                console.log('⚠️  No hay markers para mostrar');
            }}
            
            // In-price indicators (API v4)
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
                },
                grid: {
                    vertLines: { color: 'rgba(197, 203, 206, 0.1)' },
                    horzLines: { color: 'rgba(197, 203, 206, 0.1)' },
                },
                timeScale: { timeVisible: true, secondsVisible: false },
            });
            
            // Off-price indicators (API v4)
            indicatorsOff.forEach(ind => {
                const line = subChart.addLineSeries({
                    color: ind.color,
                    lineWidth: 2,
                    title: ind.name,
                });
                line.setData(ind.data);
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
            
            console.log('✅ Chart loaded successfully!');
        }}).catch(error => {{
            console.error('❌ Error loading data:', error);
        }});
    </script>
</body>
</html>'''
    
    # Guardar HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n💾 Guardado: {output_path}\n")
    
    # Servir y abrir
    serve_and_open(output_path)
    
    return output_path


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
    print(f"🌐 Servidor: http://localhost:{port}")
    print(f"{'='*60}\n")
    
    # Abrir navegador (WSL -> Windows)
    try:
        subprocess.Popen(['cmd.exe', '/c', 'start', url], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        print(f"✓ Navegador abierto automáticamente")
    except:
        print(f"⚠️  Abre manualmente: {url}")
    
    print(f"\n💡 Presiona Ctrl+C para cerrar\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✓ Cerrando...")
        httpd.shutdown()
        httpd.server_close()
        print("✓ Cerrado\n")


if __name__ == "__main__":
    print("Lightweight Charts HTML Generator")