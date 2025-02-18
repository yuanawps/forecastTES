import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Streamlit app title
st.title("Triple Exponential Smoothing for Wind Speed Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read Excel file
    df = pd.read_excel(uploaded_file)

    # Rename columns to match the expected format
    df.columns = ['Month', 'Wind Speed (Actual)']
    
    # Ensure the 'Month' column is datetime and 'Wind Speed Max' column is float
    df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
    df['Wind Speed (Actual)'] = df['Wind Speed (Actual)'].astype(float)
    df.set_index('Month', inplace=True)

    # Display actual data from January 2014 to September 2024
    actual_df = df[(df.index >= '2014-01-01') & (df.index <= '2024-09-01')]

    # Filter data for predictions starting from January 2016
    prediction_df = df[df.index >= '2016-01-01']

    # Triple Exponential Smoothing (Holt-Winters)
    model = ExponentialSmoothing(prediction_df['Wind Speed (Actual)'], trend='add', seasonal='add', seasonal_periods=24)
    fit = model.fit()

    # Add fitted values (forecast from January 2016) to the combined DataFrame
    df['Forecast'] = np.nan
    df.loc[fit.fittedvalues.index, 'Forecast'] = fit.fittedvalues

    # Create forecasts for the next 12 months (October 2024 to September 2025)
    forecast_period = pd.date_range(start='2024-10-01', end='2025-09-01', freq='MS')
    forecast = fit.forecast(len(forecast_period))

    forecast_df = pd.DataFrame({'Month': forecast_period, 'Forecast': forecast})
    forecast_df.set_index('Month', inplace=True)

    # Combine actual data and forecast into one DataFrame
    combined_df = pd.concat([df, forecast_df])

    # Filter combined data for display: actual from January 2014 to September 2024, forecast from January 2016 to September 2025
    display_df = combined_df[(combined_df.index >= '2014-01-01') & (combined_df.index <= '2025-09-01')]
    st.write("Actual and Forecast Data from January 2014 to September 2025")
    st.write(display_df)

    # Plotting with Plotly
    fig = go.Figure()

    # Add actual data trace
    fig.add_trace(go.Scatter(
        x=actual_df.index,
        y=actual_df['Wind Speed (Actual)'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue'),
        hovertemplate='Date: %{x}<br>Actual: %{y:.2f}<extra></extra>'
    ))

    # Add forecast data trace
    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=combined_df['Forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash'),
        hovertemplate='Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>'
    ))

    # Update layout for better interactivity
    fig.update_layout(
        title="Plot Triple Exponential Smoothing",
        xaxis_title="Year",
        yaxis_title="Parameter",
        hovermode="x",
        template="plotly_white",
        autosize=True
    )

     # Error metrics (only for the fitted part, excluding future forecast)
    prediction_df['Forecast'] = fit.fittedvalues
    prediction_df['Error'] = prediction_df['Wind Speed (Actual)'] - prediction_df['Forecast']
    prediction_df['Absolute Error'] = prediction_df['Error'].abs()
    prediction_df['Squared Error'] = prediction_df['Error'] ** 2
    prediction_df['Absolute Percentage Error'] = (prediction_df['Absolute Error'] / prediction_df['Wind Speed (Actual)']) * 100

    MAE = prediction_df['Absolute Error'].mean()
    RMSE = np.sqrt(prediction_df['Squared Error'].mean())
    MAPE = prediction_df['Absolute Percentage Error'].mean()

    # Display metrics
    st.write("Triple Exponential Smoothing (Holt-Winters) Results")
    st.write(f"MAE: {MAE:.2f}")
    st.write(f"RMSE: {RMSE:.2f}")
    st.write(f"MAPE: {MAPE:.2f}%")

    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Please upload an Excel file.")
