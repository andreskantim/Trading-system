import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def calculate_atr(high, low, close, period):
    """Calculate Average True Range (ATR)"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def hawkes_process(data: pd.Series, kappa: float):
    assert(kappa > 0.0)
    alpha = np.exp(-kappa)
    arr = data.to_numpy()
    output = np.zeros(len(data))
    output[:] = np.nan
    for i in range(1, len(data)):
        if np.isnan(output[i - 1]):
            output[i] = arr[i]
        else:
            output[i] = output[i - 1] * alpha + arr[i]
    return pd.Series(output, index=data.index) * kappa

def vol_signal(close: pd.Series, vol_hawkes: pd.Series, lookback:int):
    signal = np.zeros(len(close))
    q05 = vol_hawkes.rolling(lookback).quantile(0.05)
    q95 = vol_hawkes.rolling(lookback).quantile(0.95)

    last_below = -1
    curr_sig = 0

    for i in range(len(signal)):
        if vol_hawkes.iloc[i] < q05.iloc[i]:
            last_below = i
            curr_sig = 0

        if vol_hawkes.iloc[i] > q95.iloc[i] \
           and vol_hawkes.iloc[i - 1] <= q95.iloc[i - 1] \
           and last_below > 0 :

            change = close.iloc[i] - close.iloc[last_below]
            if change > 0.0:
                curr_sig = 1
            else:
                curr_sig = -1
        signal[i] = curr_sig

    return signal

def get_trades_from_signal(data: pd.DataFrame, signal: np.array):
    long_trades = []
    short_trades = []

    close_arr = data['close'].to_numpy()
    last_sig = 0.0
    open_trade = None
    idx = data.index
    for i in range(len(data)):
        if signal[i] == 1.0 and last_sig != 1.0:
            if open_trade is not None:
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                short_trades.append(open_trade)
            open_trade = [idx[i], close_arr[i], -1, np.nan]

        if signal[i] == -1.0  and last_sig != -1.0:
            if open_trade is not None:
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                long_trades.append(open_trade)
            open_trade = [idx[i], close_arr[i], -1, np.nan]

        if signal[i] == 0.0 and last_sig == -1.0:
            open_trade[2] = idx[i]
            open_trade[3] = close_arr[i]
            short_trades.append(open_trade)
            open_trade = None

        if signal[i] == 0.0  and last_sig == 1.0:
            open_trade[2] = idx[i]
            open_trade[3] = close_arr[i]
            long_trades.append(open_trade)
            open_trade = None

        last_sig = signal[i]

    long_trades = pd.DataFrame(long_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])
    short_trades = pd.DataFrame(short_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])

    if len(long_trades) > 0:
        long_trades['percent'] = (long_trades['exit_price'] - long_trades['entry_price']) / long_trades['entry_price']
        long_trades = long_trades.set_index('entry_time')
    if len(short_trades) > 0:
        short_trades['percent'] = -1 * (short_trades['exit_price'] - short_trades['entry_price']) / short_trades['entry_price']
        short_trades = short_trades.set_index('entry_time')
    return long_trades, short_trades

# Load data
print("Loading data...")
data = pd.read_csv('BTCUSDT3600.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')

# Normalize volume
norm_lookback = 336
data['atr'] = calculate_atr(np.log(data['high']), np.log(data['low']), np.log(data['close']), norm_lookback)
data['norm_range'] = (np.log(data['high']) - np.log(data['low'])) / data['atr']

# ====================================================================================
# FIGURE 1: Multiple Hawkes processes with price in LOG SCALE - SINGLE PLOT
# ====================================================================================
print("\nGenerating Figure 1: Multiple Hawkes processes (interactive)...")

kappas_to_plot = [0.01, 0.05, 0.1, 0.25, 0.5]
hawkes_dict = {}

for kappa in kappas_to_plot:
    hawkes_dict[kappa] = hawkes_process(data['norm_range'], kappa)

# Create plotly figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add price trace (log scale)
fig.add_trace(
    go.Scatter(x=data.index, y=data['close'],
               name='BTC Price',
               line=dict(color='cyan', width=2),
               hovertemplate='%{x}<br>Price: $%{y:,.0f}<extra></extra>'),
    secondary_y=False
)

# Add Hawkes processes
colors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF']
for i, kappa in enumerate(kappas_to_plot):
    fig.add_trace(
        go.Scatter(x=data.index, y=hawkes_dict[kappa],
                   name=f'κ = {kappa}',
                   line=dict(color=colors[i], width=1.5),
                   opacity=0.8,
                   hovertemplate=f'%{{x}}<br>κ={kappa}: %{{y:.3f}}<extra></extra>'),
        secondary_y=True
    )

# Update layout
fig.update_xaxes(title_text="Date", gridcolor='#333333')
fig.update_yaxes(title_text="BTC Price (USD)", type="log", secondary_y=False, gridcolor='#333333')
fig.update_yaxes(title_text="Normalized Hawkes Process", secondary_y=True, gridcolor='#333333')

fig.update_layout(
    title='Bitcoin Price (Log Scale) and Normalized Hawkes Processes',
    template='plotly_dark',
    hovermode='x unified',
    height=800,
    width=1800,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.write_html('hawkes_processes_interactive.html')
print("Saved: hawkes_processes_interactive.html")

# Also save high-res static version
plt.style.use('dark_background')
fig_static, ax1 = plt.subplots(figsize=(24, 10))

# Price in log scale
ax1.semilogy(data.index, data['close'], color='cyan', linewidth=2, label='BTC Price', zorder=5)
ax1.set_xlabel('Date', fontsize=14)
ax1.set_ylabel('BTC Price (USD) - Log Scale', fontsize=14, color='cyan')
ax1.tick_params(axis='y', labelcolor='cyan')
ax1.grid(True, alpha=0.3)

# Hawkes on secondary axis
ax2 = ax1.twinx()
colors_mpl = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF']
for i, kappa in enumerate(kappas_to_plot):
    ax2.plot(data.index, hawkes_dict[kappa],
             color=colors_mpl[i], linewidth=1.5, alpha=0.8,
             label=f'κ = {kappa}')

ax2.set_ylabel('Normalized Hawkes Process', fontsize=14)
ax2.legend(loc='upper left', fontsize=12, ncol=5)

plt.title('Bitcoin Price (Log Scale) and Normalized Hawkes Processes', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('hawkes_processes_single_plot.png', dpi=300, bbox_inches='tight')
print("Saved: hawkes_processes_single_plot.png (high-res)")
plt.close()

# ====================================================================================
# FIGURE 2: Strategy visualization - INTERACTIVE
# ====================================================================================
print("\nGenerating Figure 2: Strategy visualization (interactive)...")

selected_kappa = 0.1
selected_lookback = 168

data['v_hawk'] = hawkes_process(data['norm_range'], selected_kappa)
data['sig'] = vol_signal(data['close'], data['v_hawk'], selected_lookback)

# Get quantiles
q05 = data['v_hawk'].rolling(selected_lookback).quantile(0.05)
q95 = data['v_hawk'].rolling(selected_lookback).quantile(0.95)

# Get trades
long_trades, short_trades = get_trades_from_signal(data, data['sig'].to_numpy())

# Create subplots
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.5, 0.25, 0.25],
    subplot_titles=(
        f'Price and Trade Signals (κ={selected_kappa}, Lookback={selected_lookback})',
        'Hawkes Process with Thresholds',
        'Position Signal'
    )
)

# Row 1: Price with trade markers
fig.add_trace(
    go.Scatter(x=data.index, y=data['close'],
               name='BTC Price',
               line=dict(color='cyan', width=2),
               hovertemplate='%{x}<br>Price: $%{y:,.0f}<extra></extra>'),
    row=1, col=1
)

# Add long entries/exits
if len(long_trades) > 0:
    fig.add_trace(
        go.Scatter(x=long_trades.index, y=long_trades['entry_price'],
                   mode='markers',
                   name='Long Entry',
                   marker=dict(symbol='triangle-up', size=12, color='lime'),
                   hovertemplate='%{x}<br>Long Entry: $%{y:,.0f}<extra></extra>'),
        row=1, col=1
    )
    exits = long_trades[long_trades['exit_time'].notna()]
    if len(exits) > 0:
        fig.add_trace(
            go.Scatter(x=exits['exit_time'], y=exits['exit_price'],
                       mode='markers',
                       name='Long Exit',
                       marker=dict(symbol='triangle-down', size=12, color='red'),
                       hovertemplate='%{x}<br>Long Exit: $%{y:,.0f}<extra></extra>'),
            row=1, col=1
        )

# Add short entries/exits
if len(short_trades) > 0:
    fig.add_trace(
        go.Scatter(x=short_trades.index, y=short_trades['entry_price'],
                   mode='markers',
                   name='Short Entry',
                   marker=dict(symbol='triangle-down', size=12, color='red'),
                   hovertemplate='%{x}<br>Short Entry: $%{y:,.0f}<extra></extra>'),
        row=1, col=1
    )
    exits = short_trades[short_trades['exit_time'].notna()]
    if len(exits) > 0:
        fig.add_trace(
            go.Scatter(x=exits['exit_time'], y=exits['exit_price'],
                       mode='markers',
                       name='Short Exit',
                       marker=dict(symbol='triangle-up', size=12, color='lime'),
                       hovertemplate='%{x}<br>Short Exit: $%{y:,.0f}<extra></extra>'),
            row=1, col=1
        )

# Row 2: Hawkes with thresholds
fig.add_trace(
    go.Scatter(x=data.index, y=data['v_hawk'],
               name='Hawkes Process',
               line=dict(color='yellow', width=2),
               hovertemplate='%{x}<br>Hawkes: %{y:.3f}<extra></extra>'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=data.index, y=q05,
               name='Q05 Threshold',
               line=dict(color='green', width=1, dash='dash'),
               hovertemplate='%{x}<br>Q05: %{y:.3f}<extra></extra>'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=data.index, y=q95,
               name='Q95 Threshold',
               line=dict(color='red', width=1, dash='dash'),
               hovertemplate='%{x}<br>Q95: %{y:.3f}<extra></extra>'),
    row=2, col=1
)

# Row 3: Signal
long_periods = data[data['sig'] > 0]
short_periods = data[data['sig'] < 0]

if len(long_periods) > 0:
    fig.add_trace(
        go.Scatter(x=long_periods.index, y=long_periods['sig'],
                   name='Long Position',
                   mode='markers',
                   marker=dict(color='lime', size=4),
                   hovertemplate='%{x}<br>Long<extra></extra>'),
        row=3, col=1
    )

if len(short_periods) > 0:
    fig.add_trace(
        go.Scatter(x=short_periods.index, y=short_periods['sig'],
                   name='Short Position',
                   mode='markers',
                   marker=dict(color='red', size=4),
                   hovertemplate='%{x}<br>Short<extra></extra>'),
        row=3, col=1
    )

# Add zero line
fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.5, row=3, col=1)

# Update layout
fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="Hawkes", row=2, col=1)
fig.update_yaxes(title_text="Position", row=3, col=1, range=[-1.5, 1.5])

fig.update_layout(
    template='plotly_dark',
    height=1200,
    width=1800,
    hovermode='x unified',
    showlegend=True
)

fig.write_html('strategy_interactive.html')
print("Saved: strategy_interactive.html")

# Calculate statistics
data['next_return'] = np.log(data['close']).diff().shift(-1)
data['signal_return'] = data['sig'] * data['next_return']
win_returns = data[data['signal_return'] > 0]['signal_return'].sum()
lose_returns = data[data['signal_return'] < 0]['signal_return'].abs().sum()
signal_pf = win_returns / lose_returns if lose_returns > 0 else np.nan

long_win_rate = len(long_trades[long_trades['percent'] > 0]) / len(long_trades) if len(long_trades) > 0 else 0
short_win_rate = len(short_trades[short_trades['percent'] > 0]) / len(short_trades) if len(short_trades) > 0 else 0
long_average = long_trades['percent'].mean() if len(long_trades) > 0 else 0
short_average = short_trades['percent'].mean() if len(short_trades) > 0 else 0
time_in_market = len(data[data['sig'] != 0.0]) / len(data)

print(f"\nStrategy Statistics (κ={selected_kappa}, Lookback={selected_lookback}):")
print(f"  Profit Factor: {signal_pf:.3f}")
print(f"  Long Win Rate: {long_win_rate:.2%}")
print(f"  Long Avg Return: {long_average:.4f}")
print(f"  Short Win Rate: {short_win_rate:.2%}")
print(f"  Short Avg Return: {short_average:.4f}")
print(f"  Time In Market: {time_in_market:.2%}")

# ====================================================================================
# FIGURE 3: Multi-dimensional heatmap (Range: 0.95 to 1.1)
# ====================================================================================
print("\nGenerating Figure 3: Multi-dimensional heatmap...")

kappa_vals = [0.01, 0.05, 0.1, 0.25, 0.5]
lookback_vals = [24, 48, 96, 168, 336, 504]
norm_lookback_vals = [168, 336, 504]

all_heatmaps = []

for norm_lb in norm_lookback_vals:
    print(f"  Computing heatmap for norm_lookback={norm_lb}...")

    temp_data = data.copy()
    temp_data['atr'] = calculate_atr(np.log(temp_data['high']), np.log(temp_data['low']),
                                      np.log(temp_data['close']), norm_lb)
    temp_data['norm_range'] = (np.log(temp_data['high']) - np.log(temp_data['low'])) / temp_data['atr']

    pf_df = pd.DataFrame(index=lookback_vals, columns=kappa_vals)

    for lb in lookback_vals:
        for k in kappa_vals:
            temp_data['v_hawk'] = hawkes_process(temp_data['norm_range'], k)
            temp_data['sig'] = vol_signal(temp_data['close'], temp_data['v_hawk'], lb)

            temp_data['next_return'] = np.log(temp_data['close']).diff().shift(-1)
            temp_data['signal_return'] = temp_data['sig'] * temp_data['next_return']
            win_returns = temp_data[temp_data['signal_return'] > 0]['signal_return'].sum()
            lose_returns = temp_data[temp_data['signal_return'] < 0]['signal_return'].abs().sum()

            signal_pf = win_returns / lose_returns if lose_returns > 0 else np.nan
            pf_df.loc[lb, k] = float(signal_pf) if not np.isnan(signal_pf) else 1.0

    all_heatmaps.append((norm_lb, pf_df.astype(float)))

# Create interactive heatmaps with Plotly
fig = make_subplots(
    rows=1, cols=len(norm_lookback_vals),
    subplot_titles=[f'Norm Lookback = {nl}' for nl in norm_lookback_vals],
    horizontal_spacing=0.1
)

for idx, (norm_lb, pf_df) in enumerate(all_heatmaps):
    heatmap = go.Heatmap(
        z=pf_df.values,
        x=[str(k) for k in kappa_vals],
        y=[str(lb) for lb in lookback_vals],
        colorscale='RdYlGn',
        zmid=1.0,
        zmin=0.95,
        zmax=1.1,
        text=pf_df.values,
        texttemplate='%{text:.3f}',
        textfont={"size": 10},

        # ✔ SOLO el último heatmap muestra barra de color
        showscale=(idx == len(norm_lookback_vals) - 1),

        # ✔ El colorbar SOLO se define cuando se muestra
        colorbar=dict(
            title="Profit Factor",
            x=1.02,
            len=0.9
        ) if idx == len(norm_lookback_vals) - 1 else None,

        hovertemplate='κ=%{x}<br>Lookback=%{y}<br>PF=%{z:.4f}<extra></extra>'
    )

    fig.add_trace(heatmap, row=1, col=idx+1)

# Update axes
for idx in range(len(norm_lookback_vals)):
    fig.update_xaxes(title_text="Hawkes Kappa (κ)", row=1, col=idx+1)
    if idx == 0:
        fig.update_yaxes(title_text="Threshold Lookback", row=1, col=idx+1)

fig.update_layout(
    title='Profit Factor Heatmap: Kappa × Lookback × Norm_Lookback (Range: 0.95-1.1)',
    template='plotly_dark',
    height=600,
    width=1800
)

fig.write_html('heatmap_interactive.html')
print("Saved: heatmap_interactive.html")

# Also create high-res static version
plt.style.use('dark_background')
fig_static, axes = plt.subplots(1, len(norm_lookback_vals), figsize=(24, 8))
if len(norm_lookback_vals) == 1:
    axes = [axes]

for idx, (norm_lb, pf_df) in enumerate(all_heatmaps):
    sns.heatmap(pf_df, annot=True, fmt='.3f', cmap='RdYlGn', center=1.0,
                ax=axes[idx], cbar_kws={'label': 'Profit Factor'},
                vmin=0.95, vmax=1.1, annot_kws={"size": 9})
    axes[idx].set_title(f'Norm Lookback = {norm_lb}', fontsize=14)
    axes[idx].set_xlabel('Hawkes Kappa (κ)', fontsize=12)
    axes[idx].set_ylabel('Threshold Lookback', fontsize=12)

plt.suptitle('Profit Factor Heatmap: Kappa × Lookback × Norm_Lookback (Range: 0.95-1.1)',
             fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('heatmap_high_res.png', dpi=300, bbox_inches='tight')
print("Saved: heatmap_high_res.png (high-res)")
plt.close()

print("\n" + "="*70)
print("All visualizations completed successfully!")
print("="*70)
print("\nGenerated files:")
print("  Interactive (HTML - zoomable):")
print("    1. hawkes_processes_interactive.html")
print("    2. strategy_interactive.html")
print("    3. heatmap_interactive.html")
print("\n  High-Resolution Static (PNG):")
print("    4. hawkes_processes_single_plot.png")
print("    5. heatmap_high_res.png")
print("\nOpen the HTML files in a web browser for interactive exploration!")
