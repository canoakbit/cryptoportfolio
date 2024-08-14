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

# Load XRP pricing data
xrp_data = pd.read_csv(f'data\\XRP-USD_{year}.csv', parse_dates=['Date'])

# Load LTC pricing data
ltc_data = pd.read_csv(f'data\\LTC-USD_{year}.csv', parse_dates=['Date'])

# Load LINK pricing data
link_data = pd.read_csv(f'data\\LINK-USD_{year}.csv', parse_dates=['Date'])

# Load daily treasury yield data
treasury_data = pd.read_csv(f'data\\daily-treasury-rates_{year}.csv', parse_dates=['Date'])
treasury_data_sorted = treasury_data.sort_values(by='Date', ascending=True)

# Forward fill missing yield data
treasury_data_sorted['5 Yr'] = treasury_data_sorted['5 Yr'].ffill()

# Calculate daily ROI for BTC
btc_data['BTC_ROI'] = btc_data['Close'].pct_change() * 100

# Calculate daily ROI for ETH
eth_data['ETH_ROI'] = eth_data['Close'].pct_change() * 100

# Calculate daily ROI for XRP
xrp_data['XRP_ROI'] = xrp_data['Close'].pct_change() * 100

# Calculate daily ROI for LTC
ltc_data['LTC_ROI'] = ltc_data['Close'].pct_change() * 100

# Calculate daily ROI for LINK
link_data['LINK_ROI'] = link_data['Close'].pct_change() * 100

# Assuming a constant modified duration for the 5-Year Treasury bond
D_mod = 4.5  # Example value for 5-Year Treasury Bond

# Calculate daily ROI for 5-Year Treasury based on yield change
treasury_data_sorted['5Yr_ROI'] = -D_mod * treasury_data_sorted['5 Yr'].diff() / 100

# Merge BTC, ETH, and treasury data on the Date column
merged_data = btc_data.merge(eth_data, on='Date').merge(link_data, on='Date').merge(ltc_data, on='Date').merge(treasury_data_sorted, on='Date')

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
    ltc_weight * merged_data['LTC_ROI'] +
    treasury_weight * merged_data['5Yr_ROI']
) / 100

# Calculate Cumulative Returns for BTC, ETH, and Mixed Portfolio
merged_data['BTC_Cumulative_Return'] = (1 + merged_data['BTC_ROI'] / 100).cumprod() - 1
merged_data['ETH_Cumulative_Return'] = (1 + merged_data['ETH_ROI'] / 100).cumprod() - 1
'''merged_data['XRP_Cumulative_Return'] = (1 + merged_data['XRP_ROI'] / 100).cumprod() - 1'''
merged_data['LTC_Cumulative_Return'] = (1 + merged_data['LTC_ROI'] / 100).cumprod() - 1
'''merged_data['LINK_Cumulative_Return'] = (1 + merged_data['LINK_ROI'] / 100).cumprod() - 1'''
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
# Assume a beta for demonstration purposes, typically calculated against a market index
portfolio_beta = 1.0  # Example beta value
treynor_ratio = (average_return - risk_free_rate) / portfolio_beta

# Print out ratios
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

# Generate a timestamp
now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")

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
plt.plot(merged_data['Date'], merged_data['BTC_Cumulative_Return'], label='Cumulative BTC Return', color='blue')
plt.plot(merged_data['Date'], merged_data['ETH_Cumulative_Return'], label='Cumulative ETH Return', color='green')
plt.plot(merged_data['Date'], merged_data['Mixed_Cumulative_Return'], label='Cumulative Mixed Portfolio Return', color='red')
plt.title(f'Cumulative Returns {year}: BTC vs ETH vs Mixed Portfolio')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'output\\cumulative_returns_{year}_{timestamp}.png')
plt.show()
