import streamlit as st
import joblib
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Streamlit app configuration
st.title("Stock Return Prediction App")
st.markdown("""
This app allows users to:
- Upload a pre-trained model.
- Select a stock for prediction.
- Input macroeconomic parameters to simulate a scenario.
- View the stock's historical chart and predicted return.
""")

# Step 2: Upload pre-trained model
model_file = st.file_uploader("Upload Pre-Trained Model (.pkl file)", type=["pkl"])
if model_file:
    model = joblib.load(model_file)
    st.success("Model loaded successfully!")

    # Step 3: Select a stock
    available_stocks = ['ITC.NS', 'TCS.NS', 'WIPRO.NS', '^NSE']
    selected_stock = st.selectbox("Select a Stock", available_stocks)

    # Step 4: Fetch stock data
    start_date = st.date_input("Start Date", value=pd.to_datetime('2023-01-01'))
    end_date = st.date_input("End Date", value=pd.to_datetime('2023-12-31'))
    stock_data = yf.download(selected_stock, start=start_date, end=end_date)['Adj Close']

    # Step 5: Calculate daily returns
    returns = stock_data.pct_change().dropna()

    # Display stock's historical chart
    st.subheader("Historical Stock Performance")
    if not stock_data.empty:
        fig, ax = plt.subplots()
        stock_data.plot(ax=ax, title=f"Historical Prices for {selected_stock}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (Adjusted Close)")
        st.pyplot(fig)
    else:
        st.warning("No data available for the selected date range.")

    # Step 6: Input macroeconomic parameters
    st.subheader("Input Macroeconomic Scenario")
    inflation_rate = st.number_input("Inflation Rate (%)", value=5.0, format="%.2f")
    interest_rate = st.number_input("Interest Rate (%)", value=3.0, format="%.2f")
    geopolitical_risk = st.slider("Geopolitical Risk (0-10)", 0, 10, 5)

    # Step 7: Prepare new scenario
    new_scenario = np.array([inflation_rate, interest_rate, geopolitical_risk]).reshape(1, -1)

    # Predict the return
    if st.button("Predict Return"):
        predicted_return = model.predict(new_scenario)[0]
        st.subheader(f"Predicted Return for {selected_stock}")
        st.write(f"{predicted_return:.2f}%")

        # Step 8: Display predicted return chart alongside historical data
        st.subheader("Predicted Return Chart")
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Plot historical data
        stock_data.plot(ax=ax[0], color="blue", title=f"Historical Prices for {selected_stock}")
        ax[0].set_ylabel("Price (Adjusted Close)")
        ax[0].set_xlabel("")

        # Plot predicted return
        ax[1].bar(['Predicted Return'], [predicted_return], color='orange')
        ax[1].set_ylabel('Return (%)')
        ax[1].set_title(f'Predicted Return for {selected_stock}')

        st.pyplot(fig)
else:
    st.warning("Please upload a valid pre-trained model file to proceed.")
