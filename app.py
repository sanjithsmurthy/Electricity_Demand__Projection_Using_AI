# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
import joblib
from datetime import datetime
import os

# Set page configuration
st.set_page_config(page_title="Load Prediction Dashboard", layout="wide")

def load_data():
    """Load prediction results and metrics"""
    try:
        predictions_df = pd.read_csv(os.path.join('results', 'predictions.csv'))
        metrics_df = pd.read_csv(os.path.join('results', 'metrics.csv'))
        
        # Convert Date column to datetime
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        
        return predictions_df, metrics_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def display_daily_comparison(date, data):
    """Display daily metrics comparison between models"""
    daily_data = data[data['Date'].dt.date == date]
    
    if daily_data.empty:
        st.warning("No data available for selected date")
        return
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Metric': ['Peak Load', 'Min Load', 'Average Load'],
        'Actual': [
            daily_data['Actual_Peak'].iloc[0],
            daily_data['Actual_Min'].iloc[0],
            daily_data['Actual_Avg'].iloc[0]
        ],
        'LSTM': [
            daily_data['LSTM_Peak'].iloc[0],
            daily_data['LSTM_Min'].iloc[0],
            daily_data['LSTM_Avg'].iloc[0]
        ],
        'Hybrid': [
            daily_data['Hybrid_Peak'].iloc[0],
            daily_data['Hybrid_Min'].iloc[0],
            daily_data['Hybrid_Avg'].iloc[0]
        ]
    })
    
    # Calculate percentage differences
    for model in ['LSTM', 'Hybrid']:
        comparison[f'{model} Error %'] = (
            (comparison[model] - comparison['Actual']) / comparison['Actual'] * 100
        ).round(2)
    
    # Format numeric columns
    for col in ['Actual', 'LSTM', 'Hybrid']:
        comparison[col] = comparison[col].round(2)
    
    # Display metrics
    st.subheader(f"Load Predictions Comparison for {date}")
    
    # Show the main comparison table
    st.write("### Model Predictions")
    st.dataframe(
        comparison.style.format({
            'Actual': '{:.2f}',
            'LSTM': '{:.2f}',
            'Hybrid': '{:.2f}',
            'LSTM Error %': '{:.2f}%',
            'Hybrid Error %': '{:.2f}%'
        }).background_gradient(
            subset=['LSTM Error %', 'Hybrid Error %'],
            cmap='RdYlGn_r'
        ),
        use_container_width=True
    )
    
    # Display model performance summary
    st.write("### Model Performance Summary")
    
    # Calculate average absolute error for each model
    lstm_mae = abs(comparison['LSTM Error %']).mean()
    hybrid_mae = abs(comparison['Hybrid Error %']).mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "LSTM Average Error",
            f"{lstm_mae:.2f}%",
            delta=f"{hybrid_mae - lstm_mae:.2f}%",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Hybrid Average Error",
            f"{hybrid_mae:.2f}%",
            delta=f"{lstm_mae - hybrid_mae:.2f}%",
            delta_color="inverse"
        )

def create_yearly_plot(data, load_type):
    """Create yearly trend plot"""
    fig = go.Figure()
    
    for model in ['Actual', 'LSTM', 'Hybrid']:
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data[f"{model}_{load_type}"],
                name=f"{model} {load_type}",
                mode='lines'
            )
        )
    
    fig.update_layout(
        title=f"{load_type} - Yearly Comparison",
        xaxis_title="Date",
        yaxis_title="Load",
        height=600
    )
    
    return fig

def display_metrics(metrics_df):
    """Display metrics comparison"""
    # Format metrics for display
    metrics_df = metrics_df.round(4)
    
    # Create styled metrics table
    st.write("### Model Performance Metrics")
    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['R2']))
    
    # Create metrics visualization
    fig = go.Figure()
    
    metrics = ['MAE', 'RMSE', 'R2']
    models = metrics_df['Model'].unique()
    load_types = metrics_df['Load_Type'].unique()
    
    for metric in metrics:
        for model in models:
            model_data = metrics_df[metrics_df['Model'] == model]
            fig.add_trace(
                go.Bar(
                    name=f"{model} - {metric}",
                    x=model_data['Load_Type'],
                    y=model_data[metric],
                    text=model_data[metric].round(3),
                    textposition='auto',
                )
            )
    
    fig.update_layout(
        title="Model Performance Comparison",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig)

def main():
    st.title("Electricity Load Prediction Dashboard")
    
    # Load data
    predictions_df, metrics_df = load_data()
    
    if predictions_df is None or metrics_df is None:
        st.error("Failed to load prediction results. Please ensure the pipeline has been run and results are available.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Daily Overview", "Yearly Overview", "Model Comparison"])
    
    if page == "Daily Overview":
        st.header("Daily Predictions")
        
        # Date selection
        min_date = predictions_df['Date'].dt.date.min()
        max_date = predictions_df['Date'].dt.date.max()
        selected_date = st.date_input(
            "Select Date",
            min_value=min_date,
            max_value=max_date,
            value=min_date
        )
        
        # Display daily comparison
        display_daily_comparison(selected_date, predictions_df)
    
    elif page == "Yearly Overview":
        st.header("Yearly Trends")
        
        # Load type selection
        load_type = st.selectbox("Select Load Type", ["Peak", "Min", "Avg"])
        
        # Create and display yearly plot
        fig = create_yearly_plot(predictions_df, load_type)
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Model Comparison
        st.header("Model Performance Analysis")
        display_metrics(metrics_df)

if __name__ == "__main__":
    main()