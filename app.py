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
- View predicted return for the selected stock.
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
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    stock_data = yf.download(selected_stock, start=start_date, end=end_date)['Adj Close']

    # Calculate daily returns
    returns = stock_data.pct_change().dropna()

    # Step 5: Input macroeconomic parameters
    st.subheader("Input Macroeconomic Scenario")
    inflation_rate = st.number_input("Inflation Rate (%)", value=5.0, format="%.2f")
    interest_rate = st.number_input("Interest Rate (%)", value=3.0, format="%.2f")
    geopolitical_risk = st.slider("Geopolitical Risk (0-10)", 0, 10, 5)

    # Step 6: Prepare new scenario
    new_scenario = np.array([inflation_rate, interest_rate, geopolitical_risk]).reshape(1, -1)

    # Predict the return
    if st.button("Predict Return"):
        predicted_return = model.predict(new_scenario)[0]
        st.subheader(f"Predicted Return for {selected_stock}")
        st.write(f"{predicted_return:.2f}%")

        # Step 7: Display chart
        st.subheader("Predicted Return Chart")
        fig, ax = plt.subplots()
        ax.bar(['Predicted Return'], [predicted_return], color='blue')
        ax.set_ylabel('Return (%)')
        ax.set_title(f'Predicted Return for {selected_stock}')
        st.pyplot(fig)
else:
    st.warning("Please upload a valid pre-trained model file to proceed.")
