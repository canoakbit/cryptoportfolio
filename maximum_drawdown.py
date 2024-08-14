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

# Calculate Portfolio Daily Returns for Mixed Portfolio
merged_data['Mixed_Portfolio_Return'] = (
    btc_weight * merged_data['BTC_ROI'] +
    eth_weight * merged_data['ETH_ROI'] +
    treasury_weight * merged_data['5Yr_ROI']
) / 100

# Calculate Cumulative Returns for BTC, ETH, and Mixed Portfolio
merged_data['BTC_Cumulative_Return'] = (1 + merged_data['BTC_ROI'] / 100).cumprod() - 1
merged_data['ETH_Cumulative_Return'] = (1 + merged_data['ETH_ROI'] / 100).cumprod() - 1
merged_data['Mixed_Cumulative_Return'] = (1 + merged_data['Mixed_Portfolio_Return']).cumprod() - 1

# Generate a timestamp
now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")

# Plot Daily Returns
plt.figure(figsize=(14, 5))
plt.plot(merged_data['Date'], merged_data['BTC_ROI'], label='Daily BTC Return', color='orange')
plt.plot(merged_data['Date'], merged_data['ETH_ROI'], label='Daily ETH Return', color='purple')
plt.plot(merged_data['Date'], merged_data['Mixed_Portfolio_Return'] * 100, label='Daily Mixed Portfolio Return', color='red')
plt.title('Daily Returns: BTC vs ETH vs Mixed Portfolio')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'output\\daily_returns_{timestamp}.jpg')
plt.show()

# Plot Cumulative Returns
plt.figure(figsize=(14, 5))
plt.plot(merged_data['Date'], merged_data['BTC_Cumulative_Return'], label='Cumulative BTC Return', color='blue')
plt.plot(merged_data['Date'], merged_data['ETH_Cumulative_Return'], label='Cumulative ETH Return', color='green')
plt.plot(merged_data['Date'], merged_data['Mixed_Cumulative_Return'], label='Cumulative Mixed Portfolio Return', color='red')
plt.title('Cumulative Returns: BTC vs ETH vs Mixed Portfolio')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'output\\cumulative_returns_{timestamp}.jpg')
plt.show()
