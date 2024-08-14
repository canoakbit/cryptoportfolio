import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load BTC pricing data
btc_data = pd.read_csv('data\\BTC-USD_2023.csv', parse_dates=['Date'])

# Load daily treasury yield data
treasury_data = pd.read_csv('data\\daily-treasury-rates_2023.csv', parse_dates=['Date'])
treasury_data_sorted = treasury_data.sort_values(by='Date', ascending=True)

# Forward fill missing yield data
treasury_data_sorted['5 Yr'] = treasury_data_sorted['5 Yr'].ffill()

# Calculate daily ROI for BTC
btc_data['BTC_ROI'] = btc_data['Close'].pct_change() * 100

# Assuming a constant modified duration for the 5-Year Treasury bond
D_mod = 4.5  # Example value for 5-Year Treasury Bond

# Calculate daily ROI for 5-Year Treasury based on yield change
treasury_data_sorted['5Yr_ROI'] = -D_mod * treasury_data_sorted['5 Yr'].diff() / 100

# Merge BTC and treasury data on the Date column
merged_data = pd.merge(btc_data, treasury_data_sorted, on='Date', how='inner')

# Initial Investment (e.g., $100)
initial_investment = 100

# Calculate Cumulative Net Profit for BTC
merged_data['BTC_Cumulative_Profit'] = initial_investment * (1 + merged_data['BTC_ROI'] / 100).cumprod()

# Calculate Cumulative Net Profit for 5-Year Treasury
merged_data['5Yr_Cumulative_Profit'] = initial_investment * (1 + merged_data['5Yr_ROI'] / 100).cumprod()

# Plot the Cumulative Net Profit for BTC and 5-Year Treasury
plt.figure(figsize=(14, 7))

# Plot BTC Cumulative Net Profit
plt.plot(merged_data['Date'], merged_data['BTC_Cumulative_Profit'], label='BTC Cumulative Net Profit', color='blue')

# Plot 5-Year Treasury Cumulative Net Profit
plt.plot(merged_data['Date'], merged_data['5Yr_Cumulative_Profit'], label='5-Year Treasury Cumulative Net Profit', color='green')

plt.title('Cumulative Net Profit: BTC vs 5-Year Treasury')
plt.xlabel('Date')
plt.ylabel('Cumulative Net Profit (USD)')
plt.legend()
plt.grid(True)

# Generate a timestamp
now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M")

# Save the plot with a timestamped filename
result_file_path = f'output\\btc_vs_5yr_treasury_cumulative_profit_{timestamp}.png'
plt.savefig(result_file_path)

# Show the plot
plt.show()

# Print the cumulative net profit for the last day
last_day_btc_profit = merged_data['BTC_Cumulative_Profit'].iloc[-1]
last_day_treasury_profit = merged_data['5Yr_Cumulative_Profit'].iloc[-1]

print(f"Cumulative Net Profit on the last day:")
print(f"BTC: ${last_day_btc_profit:.2f}")
print(f"5-Year Treasury: ${last_day_treasury_profit:.2f}")
