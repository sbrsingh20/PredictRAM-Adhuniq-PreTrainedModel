import streamlit as st
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Streamlit app configuration
st.title("Stock Return Prediction App")
st.markdown("""
This app allows users to:
- Upload a pre-trained model.
- Select multiple stocks for prediction.
- Input macroeconomic parameters to simulate a scenario.
- View the selected stocks' historical charts and predicted returns.
""")

# Step 2: Upload pre-trained model
model_file = st.file_uploader("Upload Pre-Trained Model (.pkl file)", type=["pkl"])
if model_file:
    model = joblib.load(model_file)
    st.success("Model loaded successfully!")

    # Step 3: Select stocks
    available_stocks = ['ITC.NS', 'TCS.NS', 'WIPRO.NS', '^NSEI']
    selected_stocks = st.multiselect("Select Stocks", available_stocks)

    if selected_stocks:
        # Step 4: Fetch stock data
        start_date = st.date_input("Start Date", value=pd.to_datetime('2023-01-01'))
        end_date = st.date_input("End Date", value=pd.to_datetime('2023-12-31'))

        st.subheader("Historical Stock Performance")
        for stock in selected_stocks:
            stock_data = yf.download(stock, start=start_date, end=end_date)['Adj Close']

            if not stock_data.empty:
                # Plot historical data
                fig, ax = plt.subplots()
                stock_data.plot(ax=ax, title=f"Historical Prices for {stock}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (Adjusted Close)")
                st.pyplot(fig)

                # Step 5: Calculate daily returns
                returns = stock_data.pct_change().dropna()
            else:
                st.warning(f"No data available for {stock} in the selected date range.")

        # Step 6: Input macroeconomic parameters
        st.subheader("Input Macroeconomic Scenario")
        inflation_rate = st.number_input("Inflation Rate (%)", value=5.0, format="%.2f")
        interest_rate = st.number_input("Interest Rate (%)", value=3.0, format="%.2f")
        geopolitical_risk = st.slider("Geopolitical Risk (0-10)", 0, 10, 5)

        # Step 7: Prepare new scenario
        new_scenario = np.array([inflation_rate, interest_rate, geopolitical_risk]).reshape(1, -1)

        # Predict the returns for each selected stock
        if st.button("Predict Returns"):
            st.subheader("Predicted Returns for Selected Stocks")
            for stock in selected_stocks:
                predicted_return = model.predict(new_scenario)[0]
                st.write(f"**{stock}**: {predicted_return:.2f}%")

                # Step 8: Display predicted return chart
                st.subheader(f"Predicted Return Chart for {stock}")
                fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

                # Fetch stock data again for the current stock
                stock_data = yf.download(stock, start=start_date, end=end_date)['Adj Close']

                # Plot historical data
                if not stock_data.empty:
                    stock_data.plot(ax=ax[0], color="blue", title=f"Historical Prices for {stock}")
                    ax[0].set_ylabel("Price (Adjusted Close)")
                    ax[0].set_xlabel("")

                # Plot predicted return
                ax[1].bar(['Predicted Return'], [predicted_return], color='orange')
                ax[1].set_ylabel('Return (%)')
                ax[1].set_title(f'Predicted Return for {stock}')

                st.pyplot(fig)
    else:
        st.warning("Please select at least one stock to proceed.")
else:
    st.warning("Please upload a valid pre-trained model file to proceed.")
