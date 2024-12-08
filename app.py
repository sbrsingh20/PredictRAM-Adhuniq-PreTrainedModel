import streamlit as st
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Stock Prediction App", layout="wide")

# Title and description
st.title("Stock Prediction App")
st.write("Upload a pre-trained model, select a stock, and input macroeconomic parameters to predict stock returns.")

# Step 1: Upload pre-trained model
uploaded_model = st.file_uploader("Upload Pre-trained Model (.pkl)", type=["pkl"])

if uploaded_model:
    try:
        # Load the uploaded model
        model = joblib.load(uploaded_model)
        st.success("Model loaded successfully!")

        # Display model parameters and accuracy details if available
        if hasattr(model, "get_params"):
            st.subheader("Model Parameters:")
            st.json(model.get_params())

        if hasattr(model, "score"):
            st.info("Note: Accuracy details are unavailable in this pre-trained model. Please upload a compatible model.")

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
else:
    st.warning("Please upload a valid pre-trained model to proceed.")
    st.stop()

# Step 2: Define available stocks
available_stocks = ['ITC.NS', 'TCS.NS', 'WIPRO.NS', '^NSE']

# Step 3: Stock selection
st.subheader("Select Stock")
selected_stock = st.selectbox("Choose a stock:", available_stocks)

# Step 4: Date range selection
st.subheader("Select Date Range for Historical Data")
start_date = st.date_input("Start Date", value=pd.to_datetime('2023-01-01'))
end_date = st.date_input("End Date", value=pd.to_datetime('2023-12-31'))

# Fetch stock data
try:
    stock_data = yf.download(selected_stock, start=start_date, end=end_date)['Adj Close']
    st.success(f"Fetched historical data for {selected_stock} from {start_date} to {end_date}.")
except Exception as e:
    st.error(f"Error fetching stock data: {e}")
    st.stop()

# Display historical data
st.subheader("Historical Stock Data")
st.line_chart(stock_data)

# Step 5: Input macroeconomic parameters
st.subheader("Input Macroeconomic Parameters")
inflation_rate = st.number_input("Inflation Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
geopolitical_risk = st.slider("Geopolitical Risk (0-10)", min_value=0, max_value=10, step=1)

# Step 6: Prepare scenario and predict
if st.button("Predict Return"):
    try:
        # Prepare the input scenario
        new_scenario = np.array([inflation_rate, interest_rate, geopolitical_risk]).reshape(1, -1)

        # Validate the model compatibility
        if not hasattr(model, "predict"):
            st.error("The uploaded model does not support prediction. Please upload a valid model.")
        else:
            # Predict the return
            predicted_return = model.predict(new_scenario)[0]
            st.subheader(f"Predicted Return for {selected_stock}")
            st.write(f"**{predicted_return:.2f}%**")

            # Step 7: Display predicted return alongside historical data
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
    except Exception as e:
        st.error(f"Prediction failed: {e}")
