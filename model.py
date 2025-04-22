import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.imputer = KNNImputer(n_neighbors=5)
        
    def load_and_preprocess_data(self, load_path, weather_path):
        try:
            # Load data with error handling
            logger.info(f"Loading data from {load_path} and {weather_path}")
            load_df = pd.read_csv(load_path)
            weather_df = pd.read_csv(weather_path)
            
            # Verify required columns
            required_load_cols = ['Date', 'TIMESLOT', 'DELHI']
            required_weather_cols = ['Date', 'MAX', 'MIN', 'AW', 'RF', 'EVP']
            
            for col in required_load_cols:
                if col not in load_df.columns:
                    raise ValueError(f"Missing required column {col} in load data")
            for col in required_weather_cols:
                if col not in weather_df.columns:
                    raise ValueError(f"Missing required column {col} in weather data")
            
            # Convert datetime columns with explicit format
            logger.info("Converting datetime columns")
            try:
                load_df['datetime'] = pd.to_datetime(
                    load_df['Date'] + ' ' + load_df['TIMESLOT'], 
                    format='%d-%m-%Y %H:%M'
                )
                weather_df['Date'] = pd.to_datetime(
                    weather_df['Date'],
                    format='%d-%m-%Y'
                )
            except ValueError as e:
                logger.error(f"Date conversion error: {str(e)}")
                raise
            
            # Aggregate load data to daily level
            logger.info("Aggregating load data to daily level")
            daily_load = load_df.groupby(load_df['datetime'].dt.date).agg({
                'DELHI': ['max', 'min', 'mean']
            }).reset_index()
            daily_load.columns = ['Date', 'Peak_Load', 'Min_Load', 'Avg_Load']
            daily_load['Date'] = pd.to_datetime(daily_load['Date'])
            
            # Generate complete date range for 2023
            if weather_df['Date'].max().year >= 2023:
                date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
                existing_dates = weather_df[weather_df['Date'].dt.year == 2023]['Date']
                missing_dates = pd.DataFrame({'Date': date_range[~date_range.isin(existing_dates)]})
                
                # For missing dates, interpolate weather data
                if not missing_dates.empty:
                    weather_2023 = weather_df[weather_df['Date'].dt.year == 2023].copy()
                    weather_2023.set_index('Date', inplace=True)
                    weather_2023 = weather_2023.reindex(date_range)
                    weather_2023.interpolate(method='linear', inplace=True)
                    weather_2023.fillna(method='ffill', inplace=True)
                    weather_2023.fillna(method='bfill', inplace=True)
                    weather_2023.reset_index(inplace=True)
                    weather_2023.rename(columns={'index': 'Date'}, inplace=True)
                    
                    # Update weather_df with interpolated data
                    weather_df = pd.concat([
                        weather_df[weather_df['Date'].dt.year != 2023],
                        weather_2023
                    ]).sort_values('Date')
            
            # Merge with weather data
            logger.info("Merging load and weather data")
            df = pd.merge(daily_load, weather_df, on='Date', how='right')
            
            # Handle missing values
            logger.info("Handling missing values")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
            
            # Feature engineering
            logger.info("Performing feature engineering")
            df = self._engineer_features(df)
            
            # Normalize numeric features
            logger.info("Normalizing features")
            numeric_features = df.select_dtypes(include=[np.number]).columns
            for col in numeric_features:
                self.scalers[col] = MinMaxScaler()
                df[col] = self.scalers[col].fit_transform(df[[col]])
            
            logger.info("Data preprocessing completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def _engineer_features(self, df):
        """Helper method for feature engineering"""
        # Basic features
        df['Temp_Diff'] = df['MAX'] - df['MIN']
        df['Rainfall_Intensity'] = df['RF'].divide(df['EVP'].replace(0, 1e-6))
        
        # Temporal features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Season'] = df['Month'].apply(self._get_season)
        
        # Rolling features with error handling
        for window in [3, 7, 30]:
            for col in ['Peak_Load', 'Min_Load', 'Avg_Load']:
                df[f'{col}_Rolling_{window}d'] = df[col].rolling(
                    window=window,
                    min_periods=1
                ).mean()
        
        # Lag features
        for lag in [1, 7]:
            for col in ['Peak_Load', 'Min_Load', 'Avg_Load']:
                df[f'{col}_Lag_{lag}d'] = df[col].shift(lag)
        
        # Fill any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _get_season(self, month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall

class LSTMModel:
    def __init__(self, sequence_length=7, n_features=None):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        
    def build_model(self):
        try:
            logger.info("Building LSTM model")
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(128, input_shape=(self.sequence_length, self.n_features), return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(3)  # Predict Peak, Min, and Avg Load
            ])
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )
            logger.info("LSTM model built successfully")
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {str(e)}")
            raise
    
    def prepare_sequences(self, data, include_partial=False):
        try:
            X, y = [], []
            features = data.drop(['Date', 'Peak_Load', 'Min_Load', 'Avg_Load'], axis=1)
            targets = data[['Peak_Load', 'Min_Load', 'Avg_Load']]
            
            # For normal sequence creation
            for i in range(len(data) - self.sequence_length):
                X.append(features.iloc[i:(i + self.sequence_length)].values)
                y.append(targets.iloc[i + self.sequence_length].values)
            
            # Handle partial sequences for prediction if requested
            if include_partial and len(data) < self.sequence_length:
                # Pad with the first available data
                pad_size = self.sequence_length - len(data)
                padded_features = pd.concat([
                    features.iloc[[0] * pad_size],
                    features
                ])
                X.append(padded_features.values)
                if len(targets) > 0:
                    y.append(targets.iloc[-1].values)
            
            return np.array(X), np.array(y) if y else None
            
        except Exception as e:
            logger.error(f"Error preparing sequences: {str(e)}")
            raise

    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        try:
            logger.info("Training LSTM model")
            
            # Add callbacks for better training
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    min_delta=1e-4
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("LSTM model training completed")
            return self.history
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            raise

class HybridModel:
    def __init__(self, sequence_length=7):
        self.sequence_length = sequence_length
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            n_jobs=-1,
            random_state=42
        )
        self.lstm_model = None
        self.selected_features = None
        
    def select_features(self, X, y, n_features=10):
        try:
            logger.info("Selecting features using Random Forest")
            self.rf_model.fit(X, y.mean(axis=1))
            
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.selected_features = feature_importance['feature'].head(n_features).tolist()
            logger.info(f"Selected {n_features} most important features")
            return self.selected_features
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            raise
    
    def build_lstm(self, n_features):
        self.lstm_model = LSTMModel(
            sequence_length=self.sequence_length,
            n_features=n_features
        )
        self.lstm_model.build_model()
    
    def train(self, data, epochs=50, batch_size=32):
        try:
            # Select features using RF
            X = data.drop(['Date', 'Peak_Load', 'Min_Load', 'Avg_Load'], axis=1)
            y = data[['Peak_Load', 'Min_Load', 'Avg_Load']]
            self.selected_features = self.select_features(X, y)
            
            # Prepare data for LSTM
            selected_data = data[['Date', 'Peak_Load', 'Min_Load', 'Avg_Load'] + self.selected_features]
            self.build_lstm(len(self.selected_features))
            
            X_lstm, y_lstm = self.lstm_model.prepare_sequences(selected_data)
            return self.lstm_model.train(X_lstm, y_lstm, epochs=epochs, batch_size=batch_size)
            
        except Exception as e:
            logger.error(f"Error training hybrid model: {str(e)}")
            raise

def create_output_directories():
    """Create necessary directories for output files"""
    directories = ['models', 'results', 'logs']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def predict_and_save_results(lstm_model, hybrid_model, preprocessor, test_data, train_data):
    """
    Make predictions and save results with handling for initial sequences
    """
    try:
        logger.info("Generating predictions")
        
        # Create a DataFrame that includes the last days of training data and all test data
        sequence_data = pd.concat([
            train_data.tail(lstm_model.sequence_length),
            test_data
        ]).reset_index(drop=True)
        
        # Initialize arrays to store predictions
        lstm_predictions = []
        hybrid_predictions = []
        
        # Get numerical feature columns for LSTM
        feature_columns = sequence_data.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in feature_columns if col not in ['Peak_Load', 'Min_Load', 'Avg_Load']]
        
        # Get hybrid feature columns
        hybrid_features = hybrid_model.selected_features
        hybrid_numeric_features = [col for col in hybrid_features if col not in ['Date']]
        
        # Generate predictions one day at a time
        for i in range(len(train_data.tail(lstm_model.sequence_length)), len(sequence_data)):
            # Prepare sequence for current prediction
            current_sequence = sequence_data.iloc[i - lstm_model.sequence_length:i]
            
            # Prepare LSTM input (numerical features only)
            X_lstm = current_sequence[feature_columns].values.astype(np.float32)
            X_lstm = X_lstm.reshape(1, lstm_model.sequence_length, len(feature_columns))
            
            # Get LSTM prediction
            lstm_pred = lstm_model.model.predict(X_lstm, verbose=0)
            lstm_predictions.append(lstm_pred[0])
            
            # Prepare Hybrid input (numerical features only)
            X_hybrid = current_sequence[hybrid_numeric_features].values.astype(np.float32)
            X_hybrid = X_hybrid.reshape(1, lstm_model.sequence_length, len(hybrid_numeric_features))
            
            # Get Hybrid prediction
            hybrid_pred = hybrid_model.lstm_model.model.predict(X_hybrid, verbose=0)
            hybrid_predictions.append(hybrid_pred[0])
        
        # Convert predictions to numpy arrays
        lstm_predictions = np.array(lstm_predictions)
        hybrid_predictions = np.array(hybrid_predictions)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Date': test_data['Date'].values,
            'Actual_Peak': test_data['Peak_Load'].values,
            'Actual_Min': test_data['Min_Load'].values,
            'Actual_Avg': test_data['Avg_Load'].values,
            'LSTM_Peak': lstm_predictions[:, 0],
            'LSTM_Min': lstm_predictions[:, 1],
            'LSTM_Avg': lstm_predictions[:, 2],
            'Hybrid_Peak': hybrid_predictions[:, 0],
            'Hybrid_Min': hybrid_predictions[:, 1],
            'Hybrid_Avg': hybrid_predictions[:, 2]
        })
        
        # Verify all arrays have the same length
        assert len(results_df) == len(test_data), \
            f"Length mismatch: results={len(results_df)}, test_data={len(test_data)}"
        
        # Inverse transform the scaled values
        for col in ['Actual_Peak', 'Actual_Min', 'Actual_Avg',
                   'LSTM_Peak', 'LSTM_Min', 'LSTM_Avg',
                   'Hybrid_Peak', 'Hybrid_Min', 'Hybrid_Avg']:
            base_col = 'Peak_Load' if 'Peak' in col else 'Min_Load' if 'Min' in col else 'Avg_Load'
            results_df[col] = preprocessor.scalers[base_col].inverse_transform(
                results_df[col].values.reshape(-1, 1)
            )
        
        # Calculate metrics
        metrics_list = []
        for model in ['LSTM', 'Hybrid']:
            for load_type in ['Peak', 'Min', 'Avg']:
                actual = results_df[f'Actual_{load_type}']
                pred = results_df[f'{model}_{load_type}']
                
                mae = mean_absolute_error(actual, pred)
                mse = mean_squared_error(actual, pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(actual, pred)
                
                metrics_list.append({
                    'Model': model,
                    'Load_Type': load_type,
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2
                })
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(metrics_list)
        
        # Save results
        results_path = 'results/predictions.csv'
        metrics_path = 'results/metrics.csv'
        results_df.to_csv(results_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Metrics saved to {metrics_path}")
        
        return results_df, metrics_df
        
    except Exception as e:
        logger.error(f"Error in prediction and saving results: {str(e)}")
        raise

def main():
    try:
        # Create output directories
        create_output_directories()
        
        # Initialize components
        preprocessor = DataPreprocessor()
        lstm_model = LSTMModel(sequence_length=7)
        hybrid_model = HybridModel(sequence_length=7)
        
        # Load and preprocess data
        logger.info("Starting data preprocessing")
        df = preprocessor.load_and_preprocess_data(
            'delhi_load_data_2022-2023_Final.csv',
            'delhi_weather_data_2022-2023_Final.csv'
        )
        
        # Split data into train and test
        train_data = df[df['Date'].dt.year < 2023].copy()
        test_data = df[df['Date'].dt.year == 2023].copy()
        
        # Train LSTM model
        logger.info("Training LSTM model")
        lstm_model.n_features = len(train_data.columns) - 4  # Excluding Date and target columns
        lstm_model.build_model()
        X_train, y_train = lstm_model.prepare_sequences(train_data)
        lstm_history = lstm_model.train(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)
        
        # Train Hybrid model
        logger.info("Training Hybrid model")
        hybrid_history = hybrid_model.train(train_data, epochs=50, batch_size=64)
        
        # Generate and save predictions
        logger.info("Generating and saving predictions")
        results_df, metrics_df = predict_and_save_results(
            lstm_model, hybrid_model, preprocessor, test_data, train_data
        )
        
        # Save models and preprocessor
        logger.info("Saving models and preprocessor")
        lstm_model.model.save('models/lstm_model.keras')
        joblib.dump(hybrid_model, 'models/hybrid_model.pkl')
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
        
        # Save training history
        pd.DataFrame(lstm_history.history).to_csv('results/lstm_training_history.csv')
        pd.DataFrame(hybrid_history.history).to_csv('results/hybrid_training_history.csv')
        
        # Print summary metrics
        logger.info("\nModel Performance Summary:")
        print("\nMetrics Summary:")
        print(metrics_df.to_string(index=False))
        
        print("\nAverage Performance by Model:")
        avg_metrics = metrics_df.groupby('Model')[['MAE', 'RMSE', 'R2']].mean()
        print(avg_metrics.to_string())
        
        logger.info("Pipeline completed successfully")
        
        return {
            'lstm_history': lstm_history,
            'hybrid_history': hybrid_history,
            'results': results_df,
            'metrics': metrics_df
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()