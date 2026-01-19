#!/usr/bin/env python3
"""
Script de prueba minimalista para lightweight charts.
Solo carga Bitcoin 2016-2020 y lo muestra en el navegador.
"""

import pandas as pd
import json
from pathlib import Path
import http.server
import socketserver
import subprocess
import socket

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "BTCUSD" / "BTCUSD3600.pq"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "prueba.html"

def find_free_port():
    """Encuentra un puerto libre."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def main():
    print("\n" + "="*60)
    print("PRUEBA VISUALIZACIÓN LIGHTWEIGHT CHARTS")
    print("="*60 + "\n")
    
    # 1. Cargar datos
    print("📂 Cargando datos...")
    df = pd.read_parquet(DATA_FILE)
    
    # Filtrar 2016-2020
    df = df[(df.index.year >= 2016) & (df.index.year < 2020)]
    print(f"✓ Cargados {len(df):,} registros (2016-2020)\n")
    
    # 2. Preparar datos para lightweight charts
    df_plot = df.reset_index()
    
    # Convertir a timestamp Unix (segundos desde epoch)
    df_plot['time'] = pd.to_datetime(df_plot['timestamp']).astype(int) // 10**9
    
    # Filtrar filas con valores null/NaN en OHLC
    df_plot = df_plot.dropna(subset=['open', 'high', 'low', 'close'])
    
    # Convertir a tipos Python nativos y asegurar que no hay infinitos
    df_plot = df_plot.replace([float('inf'), float('-inf')], float('nan')).dropna()
    
    # Eliminar duplicados por timestamp
    df_plot = df_plot.drop_duplicates(subset=['time'], keep='first')
    
    # Ordenar por tiempo (requerido por Lightweight Charts)
    df_plot = df_plot.sort_values('time')
    
    # Convertir a dict con validación estricta
    ohlc_data = []
    volume_data = []
    skipped = 0
    for idx, row in df_plot[['time', 'open', 'high', 'low', 'close', 'volume']].iterrows():
        try:
            # Validar que todos los valores son números válidos
            t = int(row['time'])
            o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
            v = float(row['volume']) if 'volume' in row and pd.notna(row['volume']) else 0
            
            # Validar que no son NaN o infinito
            if all(pd.notna([o, h, l, c])) and all(float('-inf') < x < float('inf') for x in [o, h, l, c]):
                ohlc_data.append({
                    'time': t,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c
                })
                volume_data.append({
                    'time': t,
                    'value': v,
                    'color': 'rgba(38, 166, 154, 0.5)' if c >= o else 'rgba(239, 83, 80, 0.5)'
                })
            else:
                skipped += 1
        except (ValueError, TypeError):
            skipped += 1
    
    if skipped > 0:
        print(f"⚠️  Omitidos {skipped} registros con valores inválidos")
    
    print(f"✓ Datos limpios: {len(ohlc_data):,} registros válidos")
    
    # Mostrar una muestra de los datos
    if len(ohlc_data) > 0:
        print(f"  Primer registro: {ohlc_data[0]}")
        print(f"  Último registro: {ohlc_data[-1]}")
    print()
    
    # 3. Generar HTML
    print("📝 Generando HTML y JSON...")
    
    # Guardar JSON en archivos separados
    json_file = OUTPUT_FILE.parent / "data.json"
    volume_file = OUTPUT_FILE.parent / "volume.json"
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(ohlc_data, f, ensure_ascii=False, allow_nan=False)
    
    with open(volume_file, 'w', encoding='utf-8') as f:
        json.dump(volume_data, f, ensure_ascii=False, allow_nan=False)
    
    print(f"✓ OHLC guardado: {len(ohlc_data):,} registros → {json_file.name}")
    print(f"✓ Volumen guardado: {len(volume_data):,} registros → {volume_file.name}")
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Bitcoin 2016-2020</title>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: sans-serif;
            background: #0d1117;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        #main-chart {{
            flex: 7;
        }}
        #volume-chart {{
            flex: 3;
            border-top: 1px solid #30363d;
        }}
    </style>
</head>
<body>
    <div id="main-chart"></div>
    <div id="volume-chart"></div>
    <script>
        // Cargar ambos archivos JSON
        Promise.all([
            fetch('data.json').then(r => r.json()),
            fetch('volume.json').then(r => r.json())
        ]).then(([ohlcData, volumeData]) => {{
            console.log('OHLC data points:', ohlcData.length);
            console.log('Volume data points:', volumeData.length);
            
            // Chart principal
            const mainChart = LightweightCharts.createChart(document.getElementById('main-chart'), {{
                width: window.innerWidth,
                height: window.innerHeight * 0.7,
                layout: {{
                    background: {{ color: '#0d1117' }},
                    textColor: '#c9d1d9',
                }},
                grid: {{
                    vertLines: {{ color: 'rgba(197, 203, 206, 0.1)' }},
                    horzLines: {{ color: 'rgba(197, 203, 206, 0.1)' }},
                }},
                timeScale: {{
                    timeVisible: true,
                    secondsVisible: false,
                }},
            }});
            
            const candlestickSeries = mainChart.addSeries(LightweightCharts.CandlestickSeries, {{
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderUpColor: '#26a69a',
                borderDownColor: '#ef5350',
                wickUpColor: '#26a69a',
                wickDownColor: '#ef5350',
            }});
            candlestickSeries.setData(ohlcData);
            
            // Chart de volumen
            const volumeChart = LightweightCharts.createChart(document.getElementById('volume-chart'), {{
                width: window.innerWidth,
                height: window.innerHeight * 0.3,
                layout: {{
                    background: {{ color: '#0d1117' }},
                    textColor: '#c9d1d9',
                }},
                grid: {{
                    vertLines: {{ color: 'rgba(197, 203, 206, 0.1)' }},
                    horzLines: {{ color: 'rgba(197, 203, 206, 0.1)' }},
                }},
                timeScale: {{
                    timeVisible: true,
                    secondsVisible: false,
                }},
            }});
            
            const volumeSeries = volumeChart.addSeries(LightweightCharts.HistogramSeries, {{
                priceFormat: {{ type: 'volume' }},
                priceScaleId: '',
            }});
            volumeSeries.setData(volumeData);
            
            // Sincronizar timeframes
            mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                if (range) volumeChart.timeScale().setVisibleLogicalRange(range);
            }});
            volumeChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                if (range) mainChart.timeScale().setVisibleLogicalRange(range);
            }});
            
            // Responsive
            window.addEventListener('resize', () => {{
                mainChart.applyOptions({{
                    width: window.innerWidth,
                    height: window.innerHeight * 0.7
                }});
                volumeChart.applyOptions({{
                    width: window.innerWidth,
                    height: window.innerHeight * 0.3
                }});
            }});
            
            console.log('✅ Charts loaded successfully!');
        }}).catch(error => {{
            console.error('❌ Error loading data:', error);
        }});
    </script>
</body>
</html>'''
    
    # Guardar
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Guardado en: {OUTPUT_FILE}\n")
    
    # 4. Servir y abrir
    port = find_free_port()
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(OUTPUT_FILE.parent), **kwargs)
        def log_message(self, format, *args):
            pass  # Silenciar logs
    
    httpd = socketserver.TCPServer(("", port), Handler)
    httpd.allow_reuse_address = True
    
    url = f"http://localhost:{port}/{OUTPUT_FILE.name}"
    
    print("="*60)
    print(f"🌐 Servidor iniciado en puerto {port}")
    print("="*60 + "\n")
    
    # Abrir navegador (WSL -> Windows)
    try:
        import subprocess
        subprocess.Popen(['cmd.exe', '/c', 'start', url], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        print(f"✓ Abriendo navegador: {url}\n")
    except Exception as e:
        print(f"⚠️  Abre manualmente: {url}\n")
    print("💡 Presiona Ctrl+C para cerrar\n")
    
    # Servir
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✓ Cerrando servidor...")
        httpd.shutdown()
        httpd.server_close()
        print("✓ Listo\n")

if __name__ == "__main__":
    main()