import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt.discrete_allocation import DiscreteAllocation

# ---- APPLICATION UI ----
st.title("📈 SI Portfolio Optimizer Apps")
st.write("Analyze and Optimize your Sustainable Investment Portfolio.")

# Input ETF ticker list
ticker_input = st.text_input("Enter your sustainable investment fund (Stock/ETF/Index) tickers. (separate with commas, min 1, max 5)", "ESGU, SUSA, ESG, VSGX")
tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]

# Validate ticker count
if len(tickers) < 1 or len(tickers) > 5:
    st.error("Please enter between 1 and 5 ETFs.")
else:
    # Input investment amount
    portfolio_value = st.number_input("Enter your investment amount (USD)", min_value=1000, value=10000, step=500)

    # Input start and end dates
    start_date = st.date_input("Select Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Select End Date", value=pd.to_datetime("2025-01-01"))

    # Button to start analysis
    analyze_button = st.button("Optimize Portfolio")

    if analyze_button:
        st.info("📡 Fetching stock price data...")
        
        # ---- FETCH DATA FROM YFINANCE ----
        data = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))["Close"]
        data = pd.DataFrame(data)
        
        # Check if any data is missing
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            st.warning(f"⚠️ Warning: {missing_values} missing values detected. Cleaning data...")
        
        # Drop missing values
        data_cleaned = data.dropna()
        
        # Validate if enough data exists
        if data_cleaned.empty:
            st.error("❌ Error: No valid data available after cleaning. Try selecting a different date range or tickers.")
        else:
            # ---- PORTFOLIO OPTIMIZATION ----
            mu = mean_historical_return(data_cleaned)
            S = CovarianceShrinkage(data_cleaned).ledoit_wolf()
            ef = EfficientFrontier(mu, S)
            raw_weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()

            # ---- DISPLAY RESULTS ----
            st.subheader("📊 Optimized Portfolio Allocation")
            st.write("Here is the optimal portfolio allocation:")
            weight_df = pd.DataFrame(list(cleaned_weights.items()), columns=["ETF", "Weight"])
            st.dataframe(weight_df)

            # ---- PORTFOLIO PERFORMANCE ----
            expected_return, volatility, sharpe_ratio = ef.portfolio_performance()
            st.write(f"✅ **Expected Annual Return:** {expected_return:.2%}")
            st.write(f"✅ **Annual Volatility:** {volatility:.2%}")
            st.write(f"✅ **Sharpe Ratio:** {sharpe_ratio:.2f}")

            # ---- DISCRETE ALLOCATION ----
            latest_prices = data_cleaned.iloc[-1]
            da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_value)
            allocation, leftover = da.greedy_portfolio()

            st.subheader("🛠 Recommended Shares to Buy")
            alloc_df = pd.DataFrame(list(allocation.items()), columns=["ETF", "Shares"])
            st.dataframe(alloc_df)

            st.write(f"💰 **Remaining Cash:** ${leftover:.2f}")

            # ---- VISUALIZATION: PIE CHART ----
            fig, ax = plt.subplots()
            ax.pie(cleaned_weights.values(), labels=cleaned_weights.keys(), autopct="%1.1f%%", startangle=140)
            ax.set_title("Portfolio Allocation")
            st.pyplot(fig)
