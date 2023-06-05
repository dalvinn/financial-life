import yfinance as yf
import fredapi
from datetime import datetime

import numpy as np
import pandas as pd

# Fetch historical data from Yahoo Finance
ticker = yf.Ticker("SPY")  # Let's use S&P 500 as a proxy for the market
hist = ticker.history(period="max")  # Fetch the last 10 years of data

# Calculate annual returns
hist["Return"] = hist["Close"].pct_change()
annual_returns = (
    hist["Return"].groupby(hist.index.year).apply(lambda x: (1 + x).prod() - 1)
)

# Fetch inflation data from FRED
fred = fredapi.Fred(
    api_key="7d557b8d855748b790c863e65eb149ed"
)  # Replace with your actual FRED API key
inflation = fred.get_series("CPIAUCNS")  # CPI Inflation Rate
inflation = inflation.resample("Y").last()  # Resample to yearly data

# First convert both series to yearly data
annual_returns.index = pd.to_datetime(annual_returns.index, format="%Y")
inflation.index = pd.to_datetime(inflation.index, format="%Y")

inflation_rate = inflation.pct_change()
start_date = annual_returns.index.min()
end_date = annual_returns.index.max()
filtered_inflation = inflation_rate[
    (inflation_rate.index >= start_date) & (inflation_rate.index <= end_date)
]
# Adjust the dates for the inflation data
adjusted_inflation = filtered_inflation.copy()
adjusted_inflation.index = adjusted_inflation.index + pd.DateOffset(days=1)

# Assign column names for clarity
annual_returns.columns = ["Return"]
adjusted_inflation.columns = ["Inflation"]

# Now you can align both data
aligned_data = pd.concat([annual_returns, adjusted_inflation], axis=1, join="inner")
aligned_data.columns = ["Return", "Inflation"]

# Calculate real returns
real_returns = (1 + aligned_data["Return"]) / (1 + aligned_data["Inflation"]) - 1


# Plot the distribution of real returns
st.pyplot(real_returns.hist(bins=20))

# Now, you can use the distribution of real returns to inform the returns in your model.
