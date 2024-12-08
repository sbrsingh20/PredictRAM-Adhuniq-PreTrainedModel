import streamlit as st
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Streamlit App Configuration
st.title("Stock Return Prediction App")
st.markdown("""
This app allows users to:
- Upload a pre-trained model.
- View the model's parameters and accuracy.
- Select multiple stocks for prediction.
- Input macroeconomic parameters to simulate a scenario.
- View historical stock performance and predicted returns.
""")

# Step 1: Upload Pre-Trained Model
model_file = st.file_uploader("Upload Pre-Trained Model (.pkl file)", type=["pkl"])
if model_file:
    try:
        # Load the model
        model_details = joblib.load(model_file)
        
        # Inspect the contents of the file
        st.subheader("Model File Contents")
        st.json(model_details if isinstance(model_details, dict) else str(model_details))
        
        # Extract model and its details
        model = model_details.get('model', None)
        model_params = model_details.get('parameters', {})
        model_accuracy = model_details.get('accuracy', 'Not available')

        if model:
            st.success("Model loaded successfully!")
            
            # Display Model Details
            st.subheader("Model Details")
            st.write("**Model Type:**", type(model).__name__)
            st.write("**Model Parameters:**", model_params if model_params else "No parameters found")
            st.write("**Model Accuracy:**", model_accuracy)

            # Step 2: Select Stocks
            available_stocks = ['ITC.NS', 'TCS.NS', 'WIPRO.NS', '^NSEI']
            selected_stocks = st.multiselect("Select Stocks", available_stocks)

            if selected_stocks:
                # Date Selection
                start_date = st.date_input("Start Date", value=pd.to_datetime('2023-01-01'))
                end_date = st.date_input("End Date", value=pd.to_datetime('2023-12-31'))

                # Display Historical Stock Performance
                st.subheader("Historical Stock Performance")
                for stock in selected_stocks:
                    stock_data = yf.download(stock, start=start_date, end=end_date)['Adj Close']

                    if not stock_data.empty:
                        # Plot Historical Data
                        fig, ax = plt.subplots()
                        stock_data.plot(ax=ax, title=f"Historical Prices for {stock}")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price (Adjusted Close)")
                        st.pyplot(fig)

                        returns = stock_data.pct_change().dropna()
                    else:
                        st.warning(f"No data available for {stock} in the selected date range.")

                # Step 3: Input Macroeconomic Parameters
                st.subheader("Input Macroeconomic Scenario")
                inflation_rate = st.number_input("Inflation Rate (%)", value=5.0, format="%.2f")
                interest_rate = st.number_input("Interest Rate (%)", value=3.0, format="%.2f")
                geopolitical_risk = st.slider("Geopolitical Risk (0-10)", 0, 10, 5)

                # Prepare New Scenario for Prediction
                new_scenario = np.array([inflation_rate, interest_rate, geopolitical_risk]).reshape(1, -1)

                # Step 4: Predict Returns
                if st.button("Predict Returns"):
                    st.subheader("Predicted Returns for Selected Stocks")
                    for stock in selected_stocks:
                        predicted_return = model.predict(new_scenario)[0]
                        st.write(f"**{stock}**: {predicted_return:.2f}%")

                        # Plot Predicted Return Chart
                        fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

                        # Fetch and Plot Stock Data Again
                        stock_data = yf.download(stock, start=start_date, end=end_date)['Adj Close']
                        if not stock_data.empty:
                            stock_data.plot(ax=ax[0], color="blue", title=f"Historical Prices for {stock}")
                            ax[0].set_ylabel("Price (Adjusted Close)")
                            ax[0].set_xlabel("")

                        ax[1].bar(['Predicted Return'], [predicted_return], color='orange')
                        ax[1].set_ylabel('Return (%)')
                        ax[1].set_title(f'Predicted Return for {stock}')

                        st.pyplot(fig)
            else:
                st.warning("Please select at least one stock to proceed.")
        else:
            st.error("No valid model found in the uploaded file.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.warning("Please upload a valid pre-trained model file to proceed.")
