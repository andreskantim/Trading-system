"""
Lightweight Charts Viewer - Versión Final

Genera HTML con TradingView Lightweight Charts usando timestamp Unix.
Compatible con WSL, abre navegador automáticamente.
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

from config.paths import BACKTEST_FIGURES, ensure_directories


def extract_trades_from_signals(signals: pd.Series, ohlc: pd.DataFrame) -> List[Dict]:
    """
    Extract individual trades from signals series for visualization.

    Args:
        signals: Series with 1 (long), -1 (short), 0 (flat)
        ohlc: DataFrame with OHLC data

    Returns:
        List of trade dicts with entry/exit info and shading data
    """
    if signals.empty:
        return []
    
    # Contar cambios de señal (número de operaciones)
    signal_changes = (signals != signals.shift()).sum()
    
    # Calcular duración en años
    time_span = (ohlc.index[-1] - ohlc.index[0]).total_seconds() / (365.25 * 24 * 3600)
    
    # Señales por año
    signals_per_year = signal_changes / time_span if time_span > 0 else 0
    
    # Si hay más de 100 señales/año, NO renderizar sombreados
    if signals_per_year > 100:
        print(f"⚠️  SOMBREADOS DESACTIVADOS: {signal_changes} señales en {time_span:.1f} años = {signals_per_year:.0f} señales/año")
        print(f"    (Se mostrarán solo las flechas de señales)")
        return []

    trades = []
    current_position = 0
    entry_idx = None
    entry_price = None

    for i, idx in enumerate(signals.index):
        sig = signals.loc[idx]

        # Skip NaN values during warmup period
        if pd.isna(sig):
            continue

        # Position change detected
        if sig != current_position:
            # Close existing position if we had one
            if current_position != 0 and entry_idx is not None:
                exit_price = ohlc.loc[idx, 'close'] if idx in ohlc.index else None
                if exit_price is not None and entry_price is not None:
                    # Calculate PnL
                    if current_position == 1:  # Long
                        pnl = exit_price - entry_price
                        is_winner = pnl > 0
                    else:  # Short
                        pnl = entry_price - exit_price
                        is_winner = pnl > 0

                    entry_ts = int(pd.Timestamp(entry_idx).value // 10**9)
                    exit_ts = int(pd.Timestamp(idx).value // 10**9)

                    # Define colors with more opacity for better visibility
                    win_color = 'rgba(38, 166, 154, 0.25)'  # Green
                    loss_color = 'rgba(239, 83, 80, 0.25)'   # Red

                    trades.append({
                        'entry_time': entry_ts,
                        'exit_time': exit_ts,
                        'entry_price': float(entry_price),
                        'exit_price': float(exit_price),
                        'direction': int(current_position),
                        'is_winner': bool(is_winner),
                        'pnl': float(pnl),
                        'color': win_color if is_winner else loss_color
                    })

            # Open new position
            if sig != 0:
                entry_idx = idx
                entry_price = ohlc.loc[idx, 'close'] if idx in ohlc.index else None
            else:
                entry_idx = None
                entry_price = None

            current_position = sig

    return trades


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
        print(f"✓ {display_name} (subchart): {len(ind_df)} puntos totales, {valid_count} válidos")
        print(f"    Rango válido: {first_valid_idx} a {last_valid_idx}")
    
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
            if prev_signal != sig_val and not pd.isna(sig_val) and not pd.isna(prev_signal):
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

    # ==========================================
    # 5b. Extract trades for shading
    # ==========================================
    trades = extract_trades_from_signals(signals, df)
    winning_trades = len([t for t in trades if t['is_winner']])
    losing_trades = len([t for t in trades if not t['is_winner']])
    print(f"✓ {len(trades)} trades ({winning_trades} winners, {losing_trades} losers)")
        
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
    trades_file = output_path.parent / 'trades.json'
    
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(ohlc_json, f, ensure_ascii=False, allow_nan=False)
    
    with open(indicators_in_file, 'w', encoding='utf-8') as f:
        json.dump(indicators_in_json, f, ensure_ascii=False, allow_nan=False)
    
    with open(indicators_off_file, 'w', encoding='utf-8') as f:
        json.dump(indicators_off_json, f, ensure_ascii=False, allow_nan=False)
    
    with open(markers_file, 'w', encoding='utf-8') as f:
        json.dump(markers, f, ensure_ascii=False, allow_nan=False)

    with open(trades_file, 'w', encoding='utf-8') as f:
        json.dump(trades, f, ensure_ascii=False, allow_nan=False)
    
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
            fetch('markers.json').then(r => r.json()),
            fetch('trades.json').then(r => r.json())
        ]).then(([ohlcData, indicatorsIn, indicatorsOff, markers, trades]) => {{
            console.log('OHLC:', ohlcData.length, 'points');
            console.log('In-price indicators:', indicatorsIn.length);
            console.log('Off-price indicators:', indicatorsOff.length);
            console.log('Markers:', markers.length);
            console.log('Trades:', trades.length);
            
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
                timeScale: {{ 
                    timeVisible: true, 
                    secondsVisible: false,
                    minBarSpacing: 0.001,
                    rightOffset: 5,
                    barSpacing: 6,
                }},
            }});

            // ===================================================================
            // TRADE SHADING - Rectangles from entry price to exit price
            // ===================================================================
            trades.forEach((trade) => {{
                const topPrice = Math.max(trade.entry_price, trade.exit_price);
                const bottomPrice = Math.min(trade.entry_price, trade.exit_price);
                
                const shadingSeries = mainChart.addHistogramSeries({{
                    color: trade.color,
                    priceFormat: {{ type: 'price' }},
                    base: bottomPrice,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false,
                    autoscaleInfoProvider: () => null,
                }});

                const shadeData = [];
                ohlcData.forEach(bar => {{
                    if (bar.time >= trade.entry_time && bar.time <= trade.exit_time) {{
                        shadeData.push({{
                            time: bar.time,
                            value: topPrice,
                            color: trade.color
                        }});
                    }}
                }});

                if (shadeData.length > 0) {{
                    shadingSeries.setData(shadeData);
                }}
            }});
            console.log('✓ Trade shading applied:', trades.length, 'trades');
            
            // Candlestick series (API v4)
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
                timeScale: { 
                    timeVisible: true, 
                    secondsVisible: false,
                    minBarSpacing: 0.001,
                    rightOffset: 5,
                    barSpacing: 6,
                },
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
                
                // SOLUCIÓN: Asegurar que el primer timestamp coincida con OHLC
                // Si el indicador tiene nulls al inicio, no hay problema porque
                // ya comparten timestamps. El issue es que LWC calcula índices diferentes.
                // Forzamos que todos usen el rango completo añadiendo punto invisible al inicio
                const paddedData = [...ind.data];
                
                // Si el primer dato del indicador es null, añadir explícitamente
                // un punto en el primer timestamp de OHLC
                if (paddedData.length > 0 && ohlcData.length > 0) {
                    const firstOhlcTime = ohlcData[0].time;
                    const firstIndTime = paddedData[0].time;
                    
                    if (firstIndTime > firstOhlcTime) {
                        // Insertar punto null al inicio para alinear índice lógico
                        paddedData.unshift({ time: firstOhlcTime, value: null });
                    }
                }
                
                line.setData(paddedData);
            });
            
            // Sync timeframes (ORIGINAL)
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