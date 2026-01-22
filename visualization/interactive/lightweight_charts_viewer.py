"""
Lightweight Charts Viewer - Versión Final

Genera HTML con TradingView Lightweight Charts usando timestamp Unix.
Compatible con WSL, abre navegador automáticamente.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
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
    
    # Compatibilidad con formato legacy: 'indicators' con 'panel'
    if 'indicators' in vis_data and (not indicators_in_price and not indicators_off_price):
        legacy_indicators = vis_data.get('indicators', {})
        for name, spec in legacy_indicators.items():
            panel = spec.get('panel', 'price')
            if panel in ['lower', 'off-price', 'subchart']:
                indicators_off_price[name] = spec
            else:
                indicators_in_price[name] = spec
    
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
        
        # Eliminar NaN directamente de la Serie
        data_clean = data.dropna()
        
        if len(data_clean) == 0:
            print(f"⚠️  {display_name}: todos NaN, omitiendo")
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
        print(f"✓ {display_name}: {len(ind_df)} puntos ({first_date} a {last_date})")
    
    # ==========================================
    # 3. Preparar indicadores off-price
    # ==========================================
    indicators_off_json = []
    for name, spec in indicators_off_price.items():
        data = spec.get('data')
        if data is None or data.empty:
            continue
        
        display_name = name.replace('_', ' ').title()
        
        # Debug ANTES de eliminar NaN
        nan_count_before = data.isna().sum()
        total_before = len(data)
        
        # Eliminar NaN directamente de la Serie
        data_clean = data.dropna()
        
        # Debug DESPUÉS de eliminar NaN
        nan_count_after = data_clean.isna().sum()
        total_after = len(data_clean)
        
        if len(data_clean) == 0:
            print(f"⚠️  {display_name}: todos NaN, omitiendo")
            continue
        
        # Convertir a timestamp Unix (ahora sin NaN)
        ind_df = pd.DataFrame({
            'time': (pd.to_datetime(data_clean.index).astype(int) // 10**9).astype(int),
            'value': data_clean.values
        })
        ind_df = ind_df.drop_duplicates(subset=['time'], keep='first')
        ind_df = ind_df.sort_values('time')
        
        # Verificar que no haya NaN en el JSON final
        nan_in_json = ind_df['value'].isna().sum()
        
        indicators_off_json.append({
            'name': display_name,
            'color': spec.get('color', 'orange'),
            'data': ind_df.to_dict('records')
        })
        
        # Debug: mostrar todo
        first_date = data_clean.index[0]
        last_date = data_clean.index[-1]
        print(f"✓ {display_name} (subchart):")
        print(f"    Antes: {total_before} puntos, {nan_count_before} NaN")
        print(f"    Después dropna: {total_after} puntos, {nan_count_after} NaN")
        print(f"    JSON final: {len(ind_df)} puntos, {nan_in_json} NaN")
        print(f"    Rango: {first_date} a {last_date}")
    
    # ==========================================
    # 4. Calcular cumulative log returns
    # ==========================================
    if not signals.empty:
        # Calcular log returns
        log_returns = np.log(df['close']).diff().shift(-1)
        
        # Multiplicar por señales
        strategy_returns = signals * log_returns
        
        # Eliminar NaN antes de acumular
        strategy_returns_clean = strategy_returns.fillna(0)
        
        # Cumulative sum
        cumulative_log = strategy_returns_clean.cumsum()
        
        # Mostrar estadísticas
        final_cumulative = cumulative_log.iloc[-1] if len(cumulative_log) > 0 else 0
        print(f"✓ Cumulative Log Returns: {final_cumulative:.6f}")
    
    # ==========================================
    # 5. Preparar marcadores de señales
    # ==========================================
    markers = []
    if not signals.empty:
        prev_signal = 0  # Empezamos asumiendo flat o primera señal
        
        # Debug: contar tipos de transiciones
        transitions = {'0->1': 0, '0->-1': 0, '1->0': 0, '-1->0': 0, '1->-1': 0, '-1->1': 0}
        
        for i, idx in enumerate(signals.index):
            sig_val = signals.loc[idx]
            
            # Obtener señal anterior (si existe)
            if i > 0:
                prev_signal = signals.iloc[i-1]
            
            # Registrar transición
            if prev_signal != sig_val:
                trans_key = f'{int(prev_signal)}->{int(sig_val)}'
                if trans_key in transitions:
                    transitions[trans_key] += 1
            
            # Long entry (cualquier transición a 1)
            if sig_val == 1 and prev_signal != 1:
                timestamp = int(pd.Timestamp(idx).value // 10**9)
                markers.append({
                    'time': timestamp,
                    'position': 'belowBar',
                    'color': '#26a69a',
                    'shape': 'arrowUp',
                    'text': 'L'
                })
            
            # Short entry (cualquier transición a -1)
            elif sig_val == -1 and prev_signal != -1:
                timestamp = int(pd.Timestamp(idx).value // 10**9)
                markers.append({
                    'time': timestamp,
                    'position': 'aboveBar',
                    'color': '#ef5350',
                    'shape': 'arrowDown',
                    'text': 'S'
                })
            
            # Exit (transición a 0 desde cualquier posición)
            elif sig_val == 0 and prev_signal != 0:
                timestamp = int(pd.Timestamp(idx).value // 10**9)
                markers.append({
                    'time': timestamp,
                    'position': 'belowBar' if prev_signal == 1 else 'aboveBar',
                    'color': '#ffa726',
                    'shape': 'circle',
                    'text': 'X'
                })
        
        print(f"✓ {len(markers)} señales")
        print(f"  Transiciones detectadas:")
        for trans, count in transitions.items():
            if count > 0:
                print(f"    {trans}: {count}")
        
        # Debug: mostrar primeras y últimas señales
        signal_values = signals.value_counts().to_dict()
        print(f"  Distribución señales: {signal_values}")
        print(f"  Primeras 10 señales: {signals.head(10).tolist()}")
        print(f"  Últimas 10 señales: {signals.tail(10).tolist()}")
        
    # ==========================================
    # 6. Guardar archivos JSON
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
    # 7. Generar HTML
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
                candleSeries.setMarkers(markers);
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
                    priceLineVisible: false,
                    lastValueVisible: true,
                });
                
                // Solo setear datos si hay datos válidos
                if (ind.data && ind.data.length > 0) {
                    line.setData(ind.data);
                    console.log('✓ Indicator', ind.name, ':', ind.data.length, 'points from', 
                               new Date(ind.data[0].time * 1000).toISOString().split('T')[0], 
                               'to', new Date(ind.data[ind.data.length-1].time * 1000).toISOString().split('T')[0]);
                }
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