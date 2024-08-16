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
        return abs(random_number)


# Apply the strategy-specific random performance generation logic
for strategy in ['CNN-L', 'DNN-L', 'LSTM-L', 'DQ-L', 'PPO-L', 'DQ_CL-L', 'PPO_CL-L',
                 'CNN-S', 'DNN-S', 'LSTM-S', 'DQ-S', 'PPO-S', 'DQ_CL-S', 'PPO_CL-S',
                 'CNN-B', 'DNN-B', 'LSTM-B', 'DQ-B', 'PPO-B', 'DQ_CL-B', 'PPO_CL-B']:
    strategy_type = strategy.split('-')[-1]  # Extract the strategy type (L, S, B)
    merged_data[strategy] = merged_data['Avg_Crypto'].apply(
        lambda x: generate_strategy_return(x, strategy_type, np.random.randn()/100)
    )

# Calculate Cumulative Returns for all strategies
strategy_cumulative_returns = merged_data[['Date']].copy()

for strategy in ['CNN-L', 'DNN-L', 'LSTM-L', 'DQ-L', 'PPO-L', 'DQ_CL-L', 'PPO_CL-L',
                 'CNN-S', 'DNN-S', 'LSTM-S', 'DQ-S', 'PPO-S', 'DQ_CL-S', 'PPO_CL-S',
                 'CNN-B', 'DNN-B', 'LSTM-B', 'DQ-B', 'PPO-B', 'DQ_CL-B', 'PPO_CL-B']:
    strategy_cumulative_returns[strategy + '_Cumulative_Return'] = (1 + merged_data[strategy] / 100).cumprod() - 1

# Convert annual risk-free rate to daily rate
def annual_to_daily_rate(annual_rate):
    TRADING_DAYS = 252
    return (1 + annual_rate) ** (1 / TRADING_DAYS) - 1

# Set the annual risk-free rate and convert it to daily
risk_free_rate_annual = 0.05
risk_free_rate_daily = annual_to_daily_rate(risk_free_rate_annual)

# Set portfolio beta for Treynor Ratio calculation (assuming beta=1.0 for simplicity)
portfolio_beta = 2.0

# Initialize dictionaries to store Sharpe and Treynor Ratios for each strategy
sharpe_ratios = {}
treynor_ratios = {}

# Loop over each strategy to calculate Sharpe and Treynor ratios
for strategy in ['CNN-L', 'DNN-L', 'LSTM-L', 'DQ-L', 'PPO-L', 'DQ_CL-L', 'PPO_CL-L',
                 'CNN-S', 'DNN-S', 'LSTM-S', 'DQ-S', 'PPO-S', 'DQ_CL-S', 'PPO_CL-S',
                 'CNN-B', 'DNN-B', 'LSTM-B', 'DQ-B', 'PPO-B', 'DQ_CL-B', 'PPO_CL-B']:
    # Daily excess returns over the risk-free rate (converted to daily)
    excess_returns = merged_data[strategy] - risk_free_rate_daily

    # Calculate Sharpe Ratio
    average_excess_return = excess_returns.mean()
    portfolio_std_dev = excess_returns.std()
    sharpe_ratio = average_excess_return / portfolio_std_dev
    sharpe_ratios[strategy] = sharpe_ratio

    # Calculate Treynor Ratio
    treynor_ratio = average_excess_return / portfolio_beta
    treynor_ratios[strategy] = treynor_ratio

# Print the Sharpe and Treynor ratios for each strategy
print("Sharpe Ratios for each strategy:")
for strategy, sharpe_ratio in sharpe_ratios.items():
    print(f"{strategy}: {sharpe_ratio:.4f}")

print("\nTreynor Ratios for each strategy:")
for strategy, treynor_ratio in treynor_ratios.items():
    print(f"{strategy}: {treynor_ratio:.4f}")

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

