import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import scipy

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
# FIGURE 1: Multiple Hawkes processes with different kappas over price chart
# ====================================================================================
print("\nGenerating Figure 1: Multiple Hawkes processes...")

kappas_to_plot = [0.01, 0.05, 0.1, 0.25, 0.5]
hawkes_dict = {}

for kappa in kappas_to_plot:
    hawkes_dict[kappa] = hawkes_process(data['norm_range'], kappa)

plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)

# Top panel: Price
ax1 = fig.add_subplot(gs[0])
ax1.plot(data.index, data['close'], color='cyan', linewidth=1.5, label='BTC Price')
ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.set_title('Bitcoin Price and Normalized Hawkes Processes (Multiple Kappa Values)', fontsize=14, pad=20)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xticklabels([])

# Bottom panel: Hawkes processes
ax2 = fig.add_subplot(gs[1], sharex=ax1)
colors = plt.cm.rainbow(np.linspace(0, 1, len(kappas_to_plot)))

for i, kappa in enumerate(kappas_to_plot):
    ax2.plot(data.index, hawkes_dict[kappa],
             color=colors[i], linewidth=1.2, alpha=0.8,
             label=f'κ = {kappa}')

ax2.set_ylabel('Normalized Hawkes Process', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.legend(loc='upper left', ncol=len(kappas_to_plot))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hawkes_processes_multiple_kappas.png', dpi=300, bbox_inches='tight')
print("Saved: hawkes_processes_multiple_kappas.png")
plt.close()

# ====================================================================================
# FIGURE 2: Strategy visualization for specific kappa and lookback
# ====================================================================================
print("\nGenerating Figure 2: Strategy visualization...")

selected_kappa = 0.1
selected_lookback = 168

data['v_hawk'] = hawkes_process(data['norm_range'], selected_kappa)
data['sig'] = vol_signal(data['close'], data['v_hawk'], selected_lookback)

# Get quantiles for visualization
q05 = data['v_hawk'].rolling(selected_lookback).quantile(0.05)
q95 = data['v_hawk'].rolling(selected_lookback).quantile(0.95)

# Get trade entries
long_trades, short_trades = get_trades_from_signal(data, data['sig'].to_numpy())

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.08)

# Top: Price with trade markers
ax1 = fig.add_subplot(gs[0])
ax1.plot(data.index, data['close'], color='cyan', linewidth=1.5, label='BTC Price')

# Mark long entries (green) and exits (red)
if len(long_trades) > 0:
    for idx, row in long_trades.iterrows():
        ax1.scatter(idx, row['entry_price'], color='lime', marker='^', s=100, zorder=5, alpha=0.7)
        if pd.notna(row['exit_time']):
            ax1.scatter(row['exit_time'], row['exit_price'], color='red', marker='v', s=100, zorder=5, alpha=0.7)

# Mark short entries (red) and exits (green)
if len(short_trades) > 0:
    for idx, row in short_trades.iterrows():
        ax1.scatter(idx, row['entry_price'], color='red', marker='v', s=100, zorder=5, alpha=0.7)
        if pd.notna(row['exit_time']):
            ax1.scatter(row['exit_time'], row['exit_price'], color='lime', marker='^', s=100, zorder=5, alpha=0.7)

ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.set_title(f'Volatility Hawkes Strategy (κ={selected_kappa}, Lookback={selected_lookback})',
              fontsize=14, pad=20)
ax1.legend(['BTC Price', 'Long Entry', 'Long Exit', 'Short Entry', 'Short Exit'], loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xticklabels([])

# Middle: Hawkes process with thresholds
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(data.index, data['v_hawk'], color='yellow', linewidth=1.5, label='Hawkes Process')
ax2.plot(data.index, q05, color='green', linewidth=1, linestyle='--', alpha=0.7, label='Q05 Threshold')
ax2.plot(data.index, q95, color='red', linewidth=1, linestyle='--', alpha=0.7, label='Q95 Threshold')
ax2.fill_between(data.index, q05, q95, alpha=0.1, color='gray')
ax2.set_ylabel('Hawkes Process', fontsize=12)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xticklabels([])

# Bottom: Signal
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.fill_between(data.index, 0, data['sig'], where=(data['sig'] > 0),
                 color='lime', alpha=0.5, label='Long')
ax3.fill_between(data.index, 0, data['sig'], where=(data['sig'] < 0),
                 color='red', alpha=0.5, label='Short')
ax3.axhline(y=0, color='white', linewidth=0.5, linestyle='-')
ax3.set_ylabel('Position', fontsize=12)
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylim(-1.5, 1.5)
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'strategy_kappa_{selected_kappa}_lookback_{selected_lookback}.png', dpi=300, bbox_inches='tight')
print(f"Saved: strategy_kappa_{selected_kappa}_lookback_{selected_lookback}.png")
plt.close()

# Calculate and print statistics
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
# FIGURE 3: Multi-dimensional heatmap (Kappa x Lookback x Norm_Lookback)
# ====================================================================================
print("\nGenerating Figure 3: Multi-dimensional heatmap...")

kappa_vals = [0.01, 0.05, 0.1, 0.25, 0.5]
lookback_vals = [24, 48, 96, 168, 336, 504]
norm_lookback_vals = [168, 336, 504]

# Create subplots for each norm_lookback
fig, axes = plt.subplots(1, len(norm_lookback_vals), figsize=(20, 6))
if len(norm_lookback_vals) == 1:
    axes = [axes]

for idx, norm_lb in enumerate(norm_lookback_vals):
    print(f"  Computing heatmap for norm_lookback={norm_lb}...")

    # Recompute ATR and normalized range with this norm_lookback
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
            pf_df.loc[lb, k] = float(signal_pf) if not np.isnan(signal_pf) else 0.0

    pf_df = pf_df.astype(float)

    # Plot heatmap
    sns.heatmap(pf_df, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0,
                ax=axes[idx], cbar_kws={'label': 'Profit Factor'},
                vmin=0.5, vmax=2.0)
    axes[idx].set_title(f'Norm Lookback = {norm_lb}', fontsize=12)
    axes[idx].set_xlabel('Hawkes Kappa (κ)', fontsize=10)
    axes[idx].set_ylabel('Threshold Lookback', fontsize=10)

plt.suptitle('Profit Factor Heatmap: Kappa × Lookback × Norm_Lookback',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('heatmap_multidimensional.png', dpi=300, bbox_inches='tight')
print("Saved: heatmap_multidimensional.png")
plt.close()

print("\n" + "="*70)
print("All visualizations completed successfully!")
print("="*70)
print("\nGenerated files:")
print("  1. hawkes_processes_multiple_kappas.png")
print(f"  2. strategy_kappa_{selected_kappa}_lookback_{selected_lookback}.png")
print("  3. heatmap_multidimensional.png")
