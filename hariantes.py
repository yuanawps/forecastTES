import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Streamlit app title
st.title("Triple Exponential Smoothing for Wind Speed Prediction")

# Display description
description = """
### Panduan Pengisian File

1. **Format File yang Diterima**
   - Hanya file dalam format **Excel (.xlsx)** yang dapat diunggah.
2. **Struktur Data Kolom**
   File Excel harus memiliki **dua kolom** dengan format berikut:
   - **Kolom 1**: `Date` → Data tanggal dalam format **MMM-YY** (contoh: Jan-14).
   - **Kolom 2**: `Actual` → Data kecepatan angin dalam angka desimal.
3. **Validasi Data**
   - Sistem akan memvalidasi format file saat diunggah.
"""

st.markdown(description)

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read Excel file
    df = pd.read_excel(uploaded_file)

    # Rename columns
    df.columns = ['Date', 'Actual']
    
    # Convert 'Date' to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
    df['Actual'] = df['Actual'].astype(float)
    df.set_index('Date', inplace=True)

    # Resample data to daily frequency
    daily_df = df.resample('D').interpolate(method='linear')

    # Filter data for modeling
    actual_df = daily_df[daily_df.index < '2024-09-01']
    prediction_df = daily_df[daily_df.index >= '2015-01-01']

    # Triple Exponential Smoothing
    model = ExponentialSmoothing(prediction_df['Actual'], trend='add', seasonal='add', seasonal_periods=365)
    fit = model.fit()

    # Add forecast to dataframe
    daily_df['Forecast'] = np.nan
    daily_df.loc[fit.fittedvalues.index, 'Forecast'] = fit.fittedvalues

    # Forecast for October 2024 to September 2025
    forecast_period = pd.date_range(start='2024-10-01', end='2025-09-30', freq='D')
    forecast = fit.forecast(len(forecast_period))
    forecast_df = pd.DataFrame({'Date': forecast_period, 'Forecast': forecast}).set_index('Date')

    # Combine data for display
    combined_df = pd.concat([daily_df, forecast_df])

    # Filter combined data
    display_df = combined_df[(combined_df.index >= '2014-01-01') & (combined_df.index <= '2025-09-30')]
    st.write("Actual and Forecast Data")
    st.write(display_df)

    # Plot data
    fig = go.Figure()

    # Actual data trace
    fig.add_trace(go.Scatter(
        x=actual_df.index,
        y=actual_df['Actual'],
        mode='lines',
        name='Actual',
        line=dict(color='blue'),
        hovertemplate='Date: %{x}<br>Actual: %{y:.2f}<extra></extra>'
    ))

    # Forecast data trace
    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=combined_df['Forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash'),
        hovertemplate='Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title="Triple Exponential Smoothing (Forecast and Actual)",
        xaxis_title="Date",
        yaxis_title="Wind Speed (m/s)",
        hovermode="x",
        template="plotly_white",
        autosize=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Calculate error metrics
    prediction_df['Forecast'] = fit.fittedvalues
    prediction_df['Error'] = prediction_df['Actual'] - prediction_df['Forecast']
    prediction_df['Absolute Error'] = prediction_df['Error'].abs()
    prediction_df['Squared Error'] = prediction_df['Error'] ** 2
    prediction_df['Absolute Percentage Error'] = (prediction_df['Absolute Error'] / prediction_df['Actual']) * 100

    RMSE = np.sqrt(prediction_df['Squared Error'].mean())
    MAPE = prediction_df['Absolute Percentage Error'].mean()

    # Display metrics
    st.write("Error Metrics")
    st.write(f"RMSE: {RMSE:.2f}")
    st.write(f"MAPE: {MAPE:.2f}%")
else:
    st.write("Please upload an Excel file.")
