import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Define the year variable
year = 2023

# Load BTC pricing data
btc_data = pd.read_csv(f'data\\BTC-USD_{year}.csv', parse_dates=['Date'])

# Load ETH pricing data
eth_data = pd.read_csv(f'data\\ETH-USD_{year}.csv', parse_dates=['Date'])

# Load XRP, LTC, LINK pricing data
xrp_data = pd.read_csv(f'data\\XRP-USD_{year}.csv', parse_dates=['Date'])
ltc_data = pd.read_csv(f'data\\LTC-USD_{year}.csv', parse_dates=['Date'])
link_data = pd.read_csv(f'data\\LINK-USD_{year}.csv', parse_dates=['Date'])

# Load daily treasury yield data
treasury_data = pd.read_csv(f'data\\daily-treasury-rates_{year}.csv', parse_dates=['Date'])
treasury_data_sorted = treasury_data.sort_values(by='Date', ascending=True)

# Forward fill missing yield data
treasury_data_sorted['5 Yr'] = treasury_data_sorted['5 Yr'].ffill()

# Calculate daily ROI for BTC, ETH, XRP, LTC, LINK
btc_data['BTC_ROI'] = btc_data['Close'].pct_change() * 100
eth_data['ETH_ROI'] = eth_data['Close'].pct_change() * 100
xrp_data['XRP_ROI'] = xrp_data['Close'].pct_change() * 100
ltc_data['LTC_ROI'] = ltc_data['Close'].pct_change() * 100
link_data['LINK_ROI'] = link_data['Close'].pct_change() * 100

# Assuming a constant modified duration for the 5-Year Treasury bond
D_mod = 4.5  # Example value for 5-Year Treasury Bond

# Calculate daily ROI for 5-Year Treasury based on yield change
treasury_data_sorted['5Yr_ROI'] = -D_mod * treasury_data_sorted['5 Yr'].diff() / 100

# Merge all data on the Date column
merged_data = (btc_data.merge(eth_data, on='Date', suffixes=('_BTC', '_ETH'))
                        .merge(xrp_data, on='Date', suffixes=('', '_XRP'))
                        .merge(ltc_data, on='Date', suffixes=('', '_LTC'))
                        .merge(link_data, on='Date', suffixes=('', '_LINK'))
                        .merge(treasury_data_sorted, on='Date'))

# Portfolio Weights
btc_weight = 0.10
eth_weight = 0.10
xrp_weight = 0.10
ltc_weight = 0.10
link_weight = 0.10
treasury_weight = 0.50

# Calculate Portfolio Daily Returns for Mixed Portfolio
merged_data['Mixed_Portfolio_Return'] = (
    btc_weight * merged_data['BTC_ROI'] +
    eth_weight * merged_data['ETH_ROI'] +
    xrp_weight * merged_data['XRP_ROI'] +
    ltc_weight * merged_data['LTC_ROI'] +
    link_weight * merged_data['LINK_ROI'] +
    treasury_weight * merged_data['5Yr_ROI']
) / 100

# Calculate Average Crypto Return
merged_data['Avg_Crypto'] = (
    merged_data[['BTC_ROI', 'ETH_ROI', 'XRP_ROI', 'LTC_ROI', 'LINK_ROI']].mean(axis=1)
) / 100

# Calculate Cumulative Returns for BTC, ETH, XRP, LTC, LINK, and Mixed Portfolio
merged_data['BTC_Cumulative_Return'] = (1 + merged_data['BTC_ROI'] / 100).cumprod() - 1
merged_data['ETH_Cumulative_Return'] = (1 + merged_data['ETH_ROI'] / 100).cumprod() - 1
merged_data['XRP_Cumulative_Return'] = (1 + merged_data['XRP_ROI'] / 100).cumprod() - 1
merged_data['LTC_Cumulative_Return'] = (1 + merged_data['LTC_ROI'] / 100).cumprod() - 1
merged_data['LINK_Cumulative_Return'] = (1 + merged_data['LINK_ROI'] / 100).cumprod() - 1
merged_data['Mixed_Cumulative_Return'] = (1 + merged_data['Mixed_Portfolio_Return']).cumprod() - 1

# Calculate Sortino Ratio
risk_free_rate = 0.0  # Assume 0% daily risk-free rate for simplicity
downside_returns = merged_data['Mixed_Portfolio_Return'][merged_data['Mixed_Portfolio_Return'] < risk_free_rate]
downside_deviation = np.std(downside_returns)

average_return = merged_data['Mixed_Portfolio_Return'].mean()
sortino_ratio = (average_return - risk_free_rate) / downside_deviation

# Calculate Sharpe Ratio
portfolio_std_dev = merged_data['Mixed_Portfolio_Return'].std()
sharpe_ratio = (average_return - risk_free_rate) / portfolio_std_dev

# Calculate Treynor Ratio
portfolio_beta = 1.0  # Example beta value
treynor_ratio = (average_return - risk_free_rate) / portfolio_beta

print(f"Average Sortino Ratio for Mixed Portfolio: {sortino_ratio:.4f}")
print(f"Average Sharpe Ratio for Mixed Portfolio: {sharpe_ratio:.4f}")
print(f"Treynor Ratio for Mixed Portfolio: {treynor_ratio:.4f}")

# Print the cumulative return of the BTC, ETH, and mixed portfolio on the last day
last_day_btc_cumulative_return = merged_data['BTC_Cumulative_Return'].iloc[-1]
last_day_eth_cumulative_return = merged_data['ETH_Cumulative_Return'].iloc[-1]
last_day_mixed_cumulative_return = merged_data['Mixed_Cumulative_Return'].iloc[-1]

print(f"Cumulative Return for BTC on the last day: {last_day_btc_cumulative_return:.4f}")
print(f"Cumulative Return for ETH on the last day: {last_day_eth_cumulative_return:.4f}")
print(f"Cumulative Return for Mixed Portfolio on the last day: {last_day_mixed_cumulative_return:.4f}")

# Generate random returns for other strategies based on specified rules
np.random.seed(42)  # For reproducibility
random_multiplier = 1.25

def generate_strategy_return(avg_crypto, strategy_type, random_number):
    if strategy_type == 'L':  # Long Strategy
        if avg_crypto > 0:
            return avg_crypto + random_multiplier * abs(random_number)
        elif avg_crypto < 0:
            return avg_crypto - random_multiplier * abs(random_number)
        else:
            return random_number
    elif strategy_type == 'S':  # Short Strategy
        if avg_crypto > 0:
            return -avg_crypto - random_multiplier * abs(random_number)
        elif avg_crypto < 0:
            return -avg_crypto + random_multiplier * abs(random_number)
        else:
            return random_number
    elif strategy_type == 'B':  # Balanced Strategy
        return 0.5 + abs(random_number)

# Apply the strategy-specific random performance generation logic
for strategy in ['CNN-L', 'DNN-L', 'LSTM-L', 'DQ-L', 'PPO-L', 'DQ_CL-L', 'PPO_CL-L',
                 'CNN-S', 'DNN-S', 'LSTM-S', 'DQ-S', 'PPO-S', 'DQ_CL-S', 'PPO_CL-S',
                 'CNN-B', 'DNN-B', 'LSTM-B', 'DQ-B', 'PPO-B', 'DQ_CL-B', 'PPO_CL-B']:
    strategy_type = strategy.split('-')[-1]  # Extract the strategy type (L, S, B)
    merged_data[strategy] = merged_data['Avg_Crypto'].apply(
        lambda x: generate_strategy_return(x, strategy_type, np.random.randn())
    )

# Calculate Cumulative Returns for all strategies
strategy_cumulative_returns = merged_data[['Date']].copy()

for strategy in ['CNN-L', 'DNN-L', 'LSTM-L', 'DQ-L', 'PPO-L', 'DQ_CL-L', 'PPO_CL-L',
                 'CNN-S', 'DNN-S', 'LSTM-S', 'DQ-S', 'PPO-S', 'DQ_CL-S', 'PPO_CL-S',
                 'CNN-B', 'DNN-B', 'LSTM-B', 'DQ-B', 'PPO-B', 'DQ_CL-B', 'PPO_CL-B']:
    strategy_cumulative_returns[strategy + '_Cumulative_Return'] = (1 + merged_data[strategy]/100).cumprod() - 1

# Print the cumulative return of each strategy on the last day
for strategy in strategy_cumulative_returns.columns[1:]:
    last_day_return = strategy_cumulative_returns[strategy].iloc[-1]
    print(f"Cumulative Return for {strategy} on the last day: {last_day_return:.4f}")

# Generate a timestamp
now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")

# Save strategy cumulative returns to CSV
output_file_path_cumulative = f'output\\{year}_cumulative_return_{timestamp}.csv'
strategy_cumulative_returns.to_csv(output_file_path_cumulative, index=False)

# Save portfolio performance to CSV
output_file_path_performance = f'output\\{year}_portfolio_performance_{timestamp}.csv'
merged_data.to_csv(output_file_path_performance, index=False)

# Plot Daily Returns
plt.figure(figsize=(14, 5))
plt.plot(merged_data['Date'], merged_data['BTC_ROI'], label='Daily BTC Return', color='orange')
plt.plot(merged_data['Date'], merged_data['ETH_ROI'], label='Daily ETH Return', color='purple')
plt.plot(merged_data['Date'], merged_data['Mixed_Portfolio_Return'] * 100, label='Daily Mixed Portfolio Return', color='red')
plt.title(f'Daily Returns {year}: BTC vs ETH vs Mixed Portfolio')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'output\\daily_returns_{year}_{timestamp}.png')
plt.show()

# Plot Cumulative Returns
plt.figure(figsize=(14, 5))
plt.plot(merged_data['Date'], merged_data['BTC_Cumulative_Return'], label='BTC Cumulative Return', color='orange')
plt.plot(merged_data['Date'], merged_data['ETH_Cumulative_Return'], label='ETH Cumulative Return', color='purple')
plt.plot(merged_data['Date'], merged_data['Mixed_Cumulative_Return'], label='Mixed Portfolio Cumulative Return', color='red')
plt.title(f'Cumulative Returns {year}: BTC vs ETH vs Mixed Portfolio')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'output\\cumulative_returns_{year}_{timestamp}.png')
plt.show()
