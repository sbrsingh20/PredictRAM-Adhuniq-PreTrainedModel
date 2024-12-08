import streamlit as st
import joblib
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Web App Title
st.title("Stock Return Prediction App")

# Step 2: Upload Pre-Trained Model
st.sidebar.header("Upload Pre-Trained Model")
uploaded_model_file = st.sidebar.file_uploader("Upload a pre-trained model (.pkl file)", type=["pkl"])

if uploaded_model_file:
    model = joblib.load(uploaded_model_file)
    st.sidebar.success("Model uploaded and loaded successfully!")
    
    # Step 3: Define the list of available stocks
    available_stocks = ['ITC.NS', 'TCS.NS', 'WIPRO.NS', '^NSE']

    # Step 4: User selects a stock
    selected_stock = st.sidebar.selectbox("Select a stock", available_stocks)

    # Step 5: Date Range Input for Stock Data
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

    # Fetch stock data
    if st.sidebar.button("Fetch Stock Data"):
        stock_data = yf.download(selected_stock, start=start_date, end=end_date)['Adj Close']
        st.subheader(f"Stock Data for {selected_stock}")
        st.line_chart(stock_data)
        
        # Step 6: Calculate daily returns
        returns = stock_data.pct_change().dropna()
        st.subheader(f"Daily Returns for {selected_stock}")
        st.line_chart(returns)

        # Step 7: Macroeconomic Inputs
        st.sidebar.header("Macroeconomic Scenario")
        inflation_rate = st.sidebar.number_input("Enter inflation rate (%)", value=3.0)
        interest_rate = st.sidebar.number_input("Enter interest rate (%)", value=2.0)
        geopolitical_risk = st.sidebar.slider("Enter geopolitical risk (0-10)", 0, 10, 5)

        # Step 8: Predict the stock return
        if st.sidebar.button("Predict Return"):
            new_scenario = np.array([inflation_rate, interest_rate, geopolitical_risk]).reshape(1, -1)
            predicted_return = model.predict(new_scenario)
            
            # Step 9: Display the Prediction
            st.subheader("Prediction Result")
            st.write(f"Predicted Return for {selected_stock} under the given scenario: **{predicted_return[0]:.2f}%**")
            
            # Step 10: Chart the Prediction
            st.subheader("Visualization of Prediction")
            fig, ax = plt.subplots()
            ax.bar(["Predicted Return"], [predicted_return[0]], color="blue")
            ax.set_ylabel("Return (%)")
            st.pyplot(fig)
else:
    st.sidebar.warning("Please upload a valid pre-trained model file to proceed.")
