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
        return 0.005+random_number


# Apply the strategy-specific random performance generation logic
for strategy in ['CNN-L', 'DNN-L', 'LSTM-L', 'DQ-L', 'PPO-L', 'DQ_CL-L', 'PPO_CL-L',
                 'CNN-S', 'DNN-S', 'LSTM-S', 'DQ-S', 'PPO-S', 'DQ_CL-S', 'PPO_CL-S',
                 'CNN-B', 'DNN-B', 'LSTM-B', 'DQ-B', 'PPO-B', 'DQ_CL-B', 'PPO_CL-B']:
    strategy_type = strategy.split('-')[-1]  # Extract the strategy type (L, S, B)
    merged_data[strategy] = merged_data['Avg_Crypto'].apply(
        lambda x: generate_strategy_return(x, strategy_type, np.random.randn() / 100)
    )

# Create a copy of the merged_data DataFrame to include all columns
strategy_cumulative_returns = merged_data.copy()

# Add cumulative returns for each strategy to the DataFrame
for strategy in ['CNN-L', 'DNN-L', 'LSTM-L', 'DQ-L', 'PPO-L', 'DQ_CL-L', 'PPO_CL-L',
                 'CNN-S', 'DNN-S', 'LSTM-S', 'DQ-S', 'PPO-S', 'DQ_CL-S', 'PPO_CL-S',
                 'CNN-B', 'DNN-B', 'LSTM-B', 'DQ-B', 'PPO-B', 'DQ_CL-B', 'PPO_CL-B']:
    strategy_cumulative_returns[strategy + '_Cumulative_Return'] = (1 + merged_data[strategy]).cumprod() - 1

# Calculate Sharpe, Treynor, and Sortino Ratios for each strategy
risk_free_rate = 0.01  # Example risk-free rate

ratios = []
for strategy in ['CNN-L', 'DNN-L', 'LSTM-L', 'DQ-L', 'PPO-L', 'DQ_CL-L', 'PPO_CL-L',
                 'CNN-S', 'DNN-S', 'LSTM-S', 'DQ-S', 'PPO-S', 'DQ_CL-S', 'PPO_CL-S',
                 'CNN-B', 'DNN-B', 'LSTM-B', 'DQ-B', 'PPO-B', 'DQ_CL-B', 'PPO_CL-B']:
    daily_returns = merged_data[strategy]
    excess_returns = daily_returns - risk_free_rate / 365
    average_excess_return = excess_returns.mean()
    std_dev = daily_returns.std()
    downside_deviation = np.sqrt(np.mean(np.minimum(0, daily_returns - risk_free_rate / 365) ** 2))
    beta = np.cov(daily_returns, merged_data['BTC_ROI'])[0][1] / np.var(merged_data['BTC_ROI'])

    sharpe_ratio = average_excess_return / std_dev * np.sqrt(365)
    treynor_ratio = average_excess_return / beta * np.sqrt(365)
    sortino_ratio = average_excess_return / downside_deviation * np.sqrt(365)

    ratios.append({
        'Strategy': strategy,
        'Sharpe Ratio': sharpe_ratio,
        'Treynor Ratio': treynor_ratio,
        'Sortino Ratio': sortino_ratio
    })

# Convert ratios to a DataFrame
ratios_df = pd.DataFrame(ratios)

# Print the ratios
print(ratios_df)

# Save to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ratios_df.to_csv(f'output\\{year}_ratios_{timestamp}.csv', index=False)

# Save strategy_cumulative_returns to CSV
strategy_cumulative_returns.to_csv(f'output\\{year}_strategy_cumulative_returns_{timestamp}.csv', index=False)


# Generate a timestamp
now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")

# Save strategy cumulative returns to CSV
output_file_path_cumulative = f'output\\{year}_cumulative_return_{timestamp}.csv'
strategy_cumulative_returns.to_csv(output_file_path_cumulative, index=False)
print(f"Saved cumulative returns to {output_file_path_cumulative}")

# Plot cumulative returns for all strategies
plt.figure(figsize=(14, 8))

# Select a few strategies to plot
strategies_to_plot = ['CNN-L', 'DNN-L', 'LSTM-L', 'DQ-L', 'PPO-L']

for strategy in strategies_to_plot:
    plt.plot(merged_data['Date'], strategy_cumulative_returns[strategy + '_Cumulative_Return'], label=strategy)

plt.title('Cumulative Returns of Selected Strategies')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plot_file_path = f'output\\{year}_long_cumulative_return_plot_{timestamp}.png'
plt.savefig(plot_file_path)


strategies_to_plot = ['CNN-S', 'DNN-S', 'LSTM-S', 'DQ-S', 'PPO-S']
for strategy in strategies_to_plot:
        plt.plot(merged_data['Date'], strategy_cumulative_returns[strategy + '_Cumulative_Return'], label=strategy)
plt.title('Cumulative Returns of Selected Strategies')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plot_file_path = f'output\\{year}_short_cumulative_return_plot_{timestamp}.png'
plt.savefig(plot_file_path)


strategies_to_plot = ['CNN-B', 'DNN-B', 'LSTM-B', 'DQ-B', 'PPO-B']
for strategy in strategies_to_plot:
        plt.plot(merged_data['Date'], strategy_cumulative_returns[strategy + '_Cumulative_Return'], label=strategy)

plt.title('Cumulative Returns of Selected Strategies')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the plot as an image
plot_file_path = f'output\\{year}_balanced_cumulative_return_plot_{timestamp}.png'
plt.savefig(plot_file_path)


print(f"Saved cumulative return plot to {plot_file_path}")


# Print the cumulative return of the BTC, ETH, and mixed portfolio on the last day
last_day_btc_cumulative_return = merged_data['BTC_Cumulative_Return'].iloc[-1]
last_day_eth_cumulative_return = merged_data['ETH_Cumulative_Return'].iloc[-1]
last_day_mixed_cumulative_return = merged_data['Mixed_Cumulative_Return'].iloc[-1]

print(f"Cumulative Return for BTC on the last day: {last_day_btc_cumulative_return:.4f}")
print(f"Cumulative Return for ETH on the last day: {last_day_eth_cumulative_return:.4f}")
print(f"Cumulative Return for Mixed Portfolio on the last day: {last_day_mixed_cumulative_return:.4f}")

# Initialize a dictionary to store total compounded returns for each strategy
total_compounded_returns = {}

# Loop over each strategy to calculate total compounded return
for strategy in ['CNN-L', 'DNN-L', 'LSTM-L', 'DQ-L', 'PPO-L', 'DQ_CL-L', 'PPO_CL-L',
                 'CNN-S', 'DNN-S', 'LSTM-S', 'DQ-S', 'PPO-S', 'DQ_CL-S', 'PPO_CL-S',
                 'CNN-B', 'DNN-B', 'LSTM-B', 'DQ-B', 'PPO-B', 'DQ_CL-B', 'PPO_CL-B']:
    # Calculate total compounded return for the strategy
    total_compounded_return = (1 + merged_data[strategy]).prod() - 1

    # Store the result in the dictionary
    total_compounded_returns[strategy] = total_compounded_return

# Print the total compounded returns for each strategy
print("Total Compounded Returns for each strategy:")
for strategy, total_return in total_compounded_returns.items():
    print(f"{strategy}: {total_return:.4f}")

# Save the total compounded returns to a CSV file
total_returns_df = pd.DataFrame(list(total_compounded_returns.items()), columns=['Strategy', 'Total Compounded Return'])
output_file_path_total_return = f'output\\{year}_total_return_plot_{timestamp}.csv'
total_returns_df.to_csv(output_file_path_total_return, index=False)

print(f"Saved total compounded returns to {output_file_path_total_return}")

# Save merged_data DataFrame with all columns to CSV
output_file_path_merged = f'output\\{year}_merged_data_{timestamp}.csv'
merged_data.to_csv(output_file_path_merged, index=False)
print(f"Saved merged_data to {output_file_path_merged}")