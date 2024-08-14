import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load BTC pricing data
btc_data = pd.read_csv('data\\BTC-USD_2023.csv', parse_dates=['Date'])

# Load daily treasury yield data
treasury_data = pd.read_csv('data\\daily-treasury-rates_2023.csv', parse_dates=['Date'])

# Calculate daily ROI for BTC
btc_data['BTC_ROI'] = btc_data['Close'].pct_change() * 100

# Check the actual changes in the 5-Year Treasury yield
treasury_data['5Yr_Change'] = treasury_data['5 Yr'].diff()

# Assuming a constant modified duration for the 5-Year Treasury bond
D_mod = 4.8  # Example value for 5-Year Treasury Bond

# Calculate daily ROI for 5-Year Treasury based on yield change
treasury_data['5Yr_ROI'] = -D_mod * treasury_data['5Yr_Change'] / 100

# Save the BTC_ROI to the BTC data file
btc_data.to_csv('data\\BTC-USD_2023_with_ROI.csv', index=False)

# Save the 5Yr_ROI to the treasury data file
treasury_data.to_csv('data\\daily-treasury-rates_2023_with_ROI.csv', index=False)

# Merge BTC and treasury data on the Date column
merged_data = pd.merge(btc_data, treasury_data, on='Date', how='inner')

# Plot the daily ROI for BTC and 5-Year Treasury
plt.figure(figsize=(14, 7))

# Plot BTC ROI
plt.plot(merged_data['Date'], merged_data['BTC_ROI'], label='BTC Daily ROI (%)', color='blue')

# Plot 5-Year Treasury ROI
plt.plot(merged_data['Date'], merged_data['5Yr_ROI'], label='5-Year Treasury Daily ROI (%)', color='green')

plt.title('BTC Daily ROI vs 5-Year Treasury Daily ROI')
plt.xlabel('Date')
plt.ylabel('Percentage (%)')
plt.legend()
plt.grid(True)

# Generate a timestamp
now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M")

# Save the plot with a timestamped filename
result_file_path = f'output\\btc_vs_5yr_treasury_roi_{timestamp}.png'
plt.savefig(result_file_path)

# Show the plot
plt.show()

# Print the first few rows of the merged_data DataFrame for inspection
print(merged_data[['Date', 'BTC_ROI', '5Yr_ROI']].head(10))
