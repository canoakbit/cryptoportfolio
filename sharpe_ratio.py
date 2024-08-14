import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load BTC pricing data
btc_data = pd.read_csv('data\\BTC-USD_2023.csv', parse_dates=['Date'])

# Load ETH pricing data
eth_data = pd.read_csv('data\\ETH-USD_2023.csv', parse_dates=['Date'])

# Load daily treasury yield data
treasury_data = pd.read_csv('data\\daily-treasury-rates_2023.csv', parse_dates=['Date'])
treasury_data_sorted = treasury_data.sort_values(by='Date', ascending=True)

# Forward fill missing yield data
treasury_data_sorted['5 Yr'] = treasury_data_sorted['5 Yr'].ffill()

# Calculate daily ROI for BTC
btc_data['BTC_ROI'] = btc_data['Close'].pct_change() * 100

# Calculate daily ROI for ETH
eth_data['ETH_ROI'] = eth_data['Close'].pct_change() * 100

# Assuming a constant modified duration for the 5-Year Treasury bond
D_mod = 4.5  # Example value for 5-Year Treasury Bond

# Calculate daily ROI for 5-Year Treasury based on yield change
treasury_data_sorted['5Yr_ROI'] = -D_mod * treasury_data_sorted['5 Yr'].diff() / 100

# Merge BTC, ETH, and treasury data on the Date column
merged_data = btc_data.merge(eth_data, on='Date').merge(treasury_data_sorted, on='Date')

# Portfolio Weights
btc_weight = 0.25
eth_weight = 0.25
treasury_weight = 0.50

# Calculate Portfolio Daily Returns
merged_data['Portfolio_Return'] = (
    btc_weight * merged_data['BTC_ROI'] +
    eth_weight * merged_data['ETH_ROI'] +
    treasury_weight * merged_data['5Yr_ROI']
) / 100

# Assuming a constant daily risk-free rate (e.g., daily yield equivalent of the 5-year Treasury)
risk_free_rate_daily = (4.5 / 100) / 365  # Example risk-free rate

# Calculate Daily Excess Return
merged_data['Excess_Return'] = merged_data['Portfolio_Return'] - risk_free_rate_daily

# Calculate Daily Sharpe Ratio
merged_data['Sharpe_Ratio'] = merged_data['Excess_Return'].expanding().mean() / merged_data['Portfolio_Return'].expanding().std()

# Plot the Daily Sharpe Ratio
plt.figure(figsize=(14, 7))
plt.plot(merged_data['Date'], merged_data['Sharpe_Ratio'], label='Daily Sharpe Ratio', color='red')
plt.title('Daily Sharpe Ratio of 25% BTC, 25% ETH, and 50% 5-Year Treasury Portfolio')
plt.xlabel('Date')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.grid(True)
plt.show()

# Calculate Cumulative Returns for BTC and ETH
merged_data['BTC_Cumulative_Return'] = (1 + merged_data['BTC_ROI'] / 100).cumprod() - 1
merged_data['ETH_Cumulative_Return'] = (1 + merged_data['ETH_ROI'] / 100).cumprod() - 1

# Plot the Daily and Cumulative Returns for 100% BTC and 100% ETH Portfolios
plt.figure(figsize=(14, 10))

# Plot Daily BTC and ETH Returns
plt.subplot(2, 1, 1)
plt.plot(merged_data['Date'], merged_data['BTC_ROI'], label='Daily BTC Return', color='orange')
plt.plot(merged_data['Date'], merged_data['ETH_ROI'], label='Daily ETH Return', color='purple')
plt.title('Daily BTC and ETH Return')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.legend()
plt.grid(True)

# Plot Cumulative BTC and ETH Returns
plt.subplot(2, 1, 2)
plt.plot(merged_data['Date'], merged_data['BTC_Cumulative_Return'], label='Cumulative BTC Return', color='blue')
plt.plot(merged_data['Date'], merged_data['ETH_Cumulative_Return'], label='Cumulative ETH Return', color='green')
plt.title('Cumulative BTC and ETH Return')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
