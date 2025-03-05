import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt.discrete_allocation import DiscreteAllocation

# ---- UI APLIKASI ----
st.title("📈 Optimal Sustainable Investment Portfolio")
st.write("Analisis dan optimasi portofolio untuk dana investasi ESG.")

# Input daftar ticker ETF ESG
ticker_input = st.text_input("Masukkan daftar ETF (pisahkan dengan koma, min 1, max 5)", "ESGU, SUSA, ESG, VSGX")
tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]

# Validasi jumlah ticker
if len(tickers) < 1 or len(tickers) > 5:
    st.error("Masukkan minimal 1 dan maksimal 5 ETF.")
else:
    # Input jumlah investasi
    portfolio_value = st.number_input("Masukkan jumlah investasi (USD)", min_value=1000, value=10000, step=500)

    # Input tanggal mulai dan akhir
    start_date = st.date_input("Pilih Tanggal Mulai", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Pilih Tanggal Akhir", value=pd.to_datetime("2025-01-01"))

    # Tombol untuk mulai analisis
    analyze_button = st.button("Optimalkan Portofolio")

    if analyze_button:
        if len(tickers) == 0:
            st.error("Pilih minimal satu dana investasi!")
        else:
            st.info("📡 Mengunduh data harga saham...")
            
            # ---- AMBIL DATA YFINANCE ----
            data = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))["Close"]
            data = pd.DataFrame(data)

            # ---- OPTIMASI PORTOFOLIO ----
            mu = mean_historical_return(data)
            S = CovarianceShrinkage(data).ledoit_wolf()
            ef = EfficientFrontier(mu, S)
            raw_weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()

            # ---- TAMPILKAN HASIL ----
            st.subheader("📊 Hasil Optimasi Portofolio")
            st.write("Berikut adalah alokasi portofolio optimal:")
            weight_df = pd.DataFrame(list(cleaned_weights.items()), columns=["ETF", "Weight"])
            st.dataframe(weight_df)

            # ---- PERFORMA PORTOFOLIO ----
            expected_return, volatility, sharpe_ratio = ef.portfolio_performance()
            st.write(f"✅ **Expected Annual Return:** {expected_return:.2%}")
            st.write(f"✅ **Annual Volatility:** {volatility:.2%}")
            st.write(f"✅ **Sharpe Ratio:** {sharpe_ratio:.2f}")

            # ---- DISCRETE ALLOCATION ----
            latest_prices = data.iloc[-1]
            da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_value)
            allocation, leftover = da.greedy_portfolio()

            st.subheader("🛠 Alokasi Saham yang Harus Dibeli")
            alloc_df = pd.DataFrame(list(allocation.items()), columns=["ETF", "Shares"])
            st.dataframe(alloc_df)

            st.write(f"💰 **Dana Tersisa:** ${leftover:.2f}")

            # ---- VISUALISASI PIE CHART ----
            fig, ax = plt.subplots()
            ax.pie(cleaned_weights.values(), labels=cleaned_weights.keys(), autopct="%1.1f%%", startangle=140)
            ax.set_title("Portofolio Allocation")
            st.pyplot(fig)
