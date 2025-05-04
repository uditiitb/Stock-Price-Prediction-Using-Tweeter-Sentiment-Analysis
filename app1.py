import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import joblib

models = joblib.load('models.pkl')
scaled = joblib.load('scaled.pkl')
company_df = joblib.load('company_df.pkl')
scalers = joblib.load('scalers.pkl')
hist_30 = joblib.load('hist_30.pkl')

# print(hist_30)

def predict_model(company):
  # global scaled_data
  ndp = scaled[company][-10:]
  last_sentiment = company_df[company]['fixed_sentiment'].iloc[-1]
  # last_sentiment
  ndp = np.vstack([ndp, [[last_sentiment/2]]])
  ndp = ndp.reshape(1, 11, 1)
  # ndp.shape
  next_day_predict=models[company].predict(ndp)
  n_predict=scalers[company].inverse_transform(next_day_predict)
    # print(n_predict[0][0])
  print(hist_30[company])
  hist_30[company].append(n_predict[0][0])
  return hist_30[company]
# next_day_predict = model.pre

st.title("Stock Price Prediction")

# Dropdown for company selection
company = st.selectbox("Select a Company", ["BAC", "C", "CSCO", "D","MSFT"])

if st.button("Show Prediction Result"):
    st.write(f"Predicting 1 days for {company}")

    # Call model_predict
    prediction = predict_model(company)
    prediction = np.array(prediction).flatten()

    # Plotting
    fig, ax = plt.subplots()

    # Plot each segment with color
    for i in range(1, len(prediction)):
        x_vals = [i, i + 1]
        y_vals = [prediction[i - 1], prediction[i]]
        color = 'green' if prediction[i] >= prediction[i - 1] else 'red'
        ax.plot(x_vals, y_vals, color=color, linewidth=2)

    ax.set_xlabel("Day")
    ax.set_ylabel("Predicted Price")
    ax.set_title(f"{company} Stock Price Prediction for {len(prediction)} Days")
    st.pyplot(fig)

