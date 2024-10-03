import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from prophet import Prophet
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
import time
import sqlite3  # Import sqlite3 module

# Suppress warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging

# Function to initialize the database
def init_db(db_name='files/Ticker_results.db'):
    # Ensure the 'files' directory exists
    os.makedirs('files', exist_ok=True)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # Create tables for performance metrics and forecasts
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            model_name TEXT,
            mae REAL,
            rmse REAL,
            mape REAL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            forecast_date TEXT,
            predicted_price REAL,
            delta REAL,
            model_names TEXT,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn, cursor

# Initialize database connection
conn, cursor = init_db()

# Set page configuration
st.set_page_config(layout="centered", page_title="Stock and ETF Price Forecasting with Multiple Models")

# Language selection moved to the top
language = st.selectbox('Choose Language / 選擇語言', ['English', '繁體中文'])

# Streamlit app title with translation
if language == 'English':
    st.title("📈 Stock and ETF Price Forecasting with Multiple Models and Ensembling")
else:
    st.title("📈 使用多種模型和集成的股票和ETF價格預測")


# Instructions with explanations
if language == 'English':
    with st.expander("Instructions", expanded=False):
        st.markdown("""
        ## Welcome to the Stock and ETF Price Forecasting App!

        This app performs time series forecasting on selected stocks and Exchange-Traded Funds (ETFs) using multiple models and techniques. It compares model performances, ensembles forecasts, and displays backtesting results to provide enhanced accuracy.

        ### **How It Works**

        - **Data Collection**: The app collects historical price data for the selected tickers.
        - **Feature Engineering**: It computes technical indicators to enrich the dataset.
        - **Model Training and Evaluation**: Multiple models are trained and evaluated using cross-validation.
        - **Forecasting**: Generates forecasts using each model and an ensemble of all models.
        - **Results Visualization**: Displays the forecasts, backtesting performance, and historical data.

        ### **Models Used**

        - **Statistical Models**: 
            - **ARIMA/SARIMA/ARIMAX**: These models capture patterns in time series data based on autoregressive and moving average components.
        - **Machine Learning Models**:
            - **XGBoost**: An efficient implementation of gradient boosting for supervised learning tasks.
        - **Time Series Models**:
            - **Facebook Prophet**: A model designed for forecasting time series data, especially with strong seasonal effects.
        - **Deep Learning Models**:
            - **LSTM Neural Networks**: Long Short-Term Memory networks are capable of learning order dependence in sequence prediction problems.

        Each model brings a unique approach to forecasting, and by comparing them, we aim to find the most accurate predictions.
        """)
else:
    with st.expander("详细说明", expanded=False):
        st.markdown("""
        ## 歡迎使用股票和ETF價格預測應用程式！

        這個應用程式使用多種模型和技術對選定的股票和交易所交易基金（ETF）進行時間序列預測。它比較模型性能，集成預測，並顯示回測結果以提供更高的準確性。

        ### **如何運作**

        - **數據收集**：應用程式收集選定標的的歷史價格數據。
        - **特徵工程**：計算技術指標以豐富數據集。
        - **模型訓練和評估**：使用交叉驗證訓練和評估多個模型。
        - **預測**：使用每個模型和所有模型的集成生成預測。
        - **結果可視化**：顯示預測、回測性能和歷史數據。

        ### **使用的模型**

        - **統計模型**：
            - **ARIMA/SARIMA/ARIMAX**：這些模型基於自回歸和移動平均組件來捕獲時間序列數據中的模式。
        - **機器學習模型**：
            - **XGBoost**：一種用於監督學習任務的高效梯度提升實現。
        - **時間序列模型**：
            - **Facebook Prophet**：專為時間序列數據預測而設計的模型，特別適用於具有強烈季節性影響的數據。
        - **深度學習模型**：
            - **LSTM神經網絡**：長短期記憶網絡能夠學習序列預測問題中的順序依賴性。

        每個模型都帶來了獨特的預測方法，通過比較它們，我們旨在找到最準確的預測。
        """)

# Debug mode
debug_mode = st.checkbox("Show Debug Information" if language == 'English' else "顯示除錯信息", value=False)

# User Options
app_mode = st.selectbox('Choose the app mode' if language == 'English' else '選擇應用模式', ['Run New Analysis' if language == 'English' else '運行新分析', 'View Saved Results' if language == 'English' else '查看已保存的結果'])

if app_mode == 'Run New Analysis' if language == 'English' else '運行新分析':
    # User Inputs
    st.header("Input Parameters" if language == 'English' else "輸入參數")

    # Ticker symbol input
    ticker_input = st.text_input(
        'Enter ticker symbols separated by commas (e.g., AAPL, GOOG, TSLA, META, MSFT, RSP, QQQ, MOAT, PFF, VNQ, IWY)' if language == 'English' else '輸入股票代碼，以逗號分隔（例如，AAPL, GOOG, TSLA, META, MSFT, RSP, QQQ, MOAT, PFF, VNQ, IWY）',
        value='TSLA, QQQ'  # Default tickers
    )

    # Process the ticker symbols
    selected_tickers = [ticker.strip().upper() for ticker in ticker_input.split(',') if ticker.strip()]

    # Check if any tickers are entered
    if not selected_tickers:
        st.error("Please enter at least one ticker symbol." if language == 'English' else "請至少輸入一個股票代碼。")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        # Date range selection
        current_date = pd.to_datetime('today').normalize()
        start_date = st.date_input(
            "Start Date" if language == 'English' else '開始日期',
            value=(current_date - pd.DateOffset(years=5)).date(),
            max_value=current_date.date()
        )
        end_date = st.date_input(
            "End Date" if language == 'English' else '結束日期',
            value=current_date.date(),
            min_value=start_date,
            max_value=current_date.date()
        )

    with col2:
        # Forecast periods (Default set to 30)
        forecast_periods = st.slider(
            'Forecast Periods (Days Ahead)' if language == 'English' else '預測期（未來幾天）',
            min_value=1,
            max_value=60,
            value=30,  # Default value set to 30
            step=1
        )
        # Rolling window size for backtesting
        rolling_window_size = st.slider(
            'Rolling Window Size for Backtesting (Days)' if language == 'English' else '回測的滾動窗口大小（天）',
            min_value=60,
            max_value=365,
            value=180,
            step=1
        )

    with st.expander("Model Parameters" if language == 'English' else "模型參數"):
        colA, colB = st.columns(2)

        with colA:
            # Model parameters with explanations
            st.markdown("**ARIMA Model Parameters**" if language == 'English' else "**ARIMA模型參數**")
            max_p = st.slider('Max p (AR order)' if language == 'English' else '最大p值（AR階數）',
                              0, 10, 5, help='Maximum number of lag observations included in the model (AutoRegressive terms).' if language == 'English' else '模型中包含的滯後觀測值的最大數量（自回歸項）。')
            max_d = st.slider('Max d (Differencing order)' if language == 'English' else '最大d值（差分階數）',
                              0, 2, 1, help='Number of times the data is differenced to make it stationary.' if language == 'English' else '對數據進行差分以使其平穩的次數。')
            max_q = st.slider('Max q (MA order)' if language == 'English' else '最大q值（MA階數）',
                              0, 10, 5, help='Maximum size of the moving average window (Moving Average terms).' if language == 'English' else '移動平均窗口的最大大小（移動平均項）。')

        with colB:
            st.markdown("**Seasonal Parameters**" if language == 'English' else "**季節性參數**")
            seasonal = st.checkbox('Include Seasonality (SARIMA)' if language == 'English' else '包含季節性（SARIMA）', value=False)
            m = st.number_input('Seasonal Period (m)' if language == 'English' else '季節性週期（m）',
                                min_value=1, value=7, step=1, help='Number of time steps for a single seasonal period (e.g., 7 for weekly seasonality).' if language == 'English' else '單個季節性週期的時間步數（例如，每週季節性為7）。')

            # Exogenous variables (for ARIMAX and Prophet)
            st.markdown("**Include Exogenous Variables (ARIMAX and Prophet)**" if language == 'English' else "**包含外生變數（ARIMAX和Prophet）**")
            use_exogenous = st.checkbox('Use Exogenous Variables' if language == 'English' else '使用外生變數', value=False)
            if use_exogenous:
                exog_input = st.text_input(
                    'Enter exogenous variables (ticker symbols) separated by commas (e.g., ^GSPC, ^IXIC, ^DJI)' if language == 'English' else '輸入外生變數（股票代碼），以逗號分隔（例如，^GSPC, ^IXIC, ^DJI）',
                    value='^GSPC, ^IXIC'
                )
                selected_exogenous = [ticker.strip().upper() for ticker in exog_input.split(',') if ticker.strip()]
            else:
                selected_exogenous = []

    # Check if start date is before end date
    if start_date >= end_date:
        st.error("Error: End date must be after start date." if language == 'English' else "錯誤：結束日期必須在開始日期之後。")
        st.stop()

    # Button to start analysis
    if st.button('Start Analysis' if language == 'English' else '開始分析'):
        with st.spinner('Running analysis...' if language == 'English' else '正在進行分析...'):
            st.markdown(f"## {'Analysis Results' if language == 'English' else '分析結果'}")

            # Initialize dictionaries
            ticker_data = {}
            latest_prices = {}
            results = {}

            # Progress bar
            progress_bar = st.progress(0)
            progress_step = 1 / len(selected_tickers) if selected_tickers else 1

            # Fetch exogenous data if needed
            if use_exogenous and selected_exogenous:
                st.write("Fetching exogenous data..." if language == 'English' else "正在獲取外生變數數據...")
                exog_data = {}
                for exog_symbol in selected_exogenous:
                    exog_df = yf.download(exog_symbol, start=start_date, end=end_date + pd.DateOffset(days=1), progress=False)
                    exog_df = exog_df['Adj Close'].resample('D').last().fillna(method='ffill')
                    exog_df = exog_df[exog_df.index <= current_date]
                    exog_data[exog_symbol] = exog_df
                exog_df_combined = pd.DataFrame(exog_data)
                exog_df_combined = exog_df_combined.fillna(method='ffill').fillna(method='bfill')
            else:
                exog_df_combined = None

            # Process each ticker
            for idx, ticker in enumerate(selected_tickers):
                st.markdown(f"### {ticker}")
                start_time_ticker = time.time()
                try:
                    data = yf.download(ticker, start=start_date, end=end_date + pd.DateOffset(days=1), progress=False)
                except Exception as e:
                    st.warning(f"Failed to fetch data for {ticker}: {e}")
                    continue

                # Check if data is empty
                if data.empty:
                    st.warning(f"No data found for {ticker}. Skipping..." if language == 'English' else f"未找到{ticker}的數據，跳過...")
                    continue

                # Use daily data
                data = data['Adj Close'].resample('D').last().fillna(method='ffill')
                data = data[data.index <= current_date]
                ticker_data[ticker] = data
                # Store the latest closing price
                latest_prices[ticker] = data.iloc[-1]

                # Ensure the data has enough points
                if len(data) < 365:  # Minimum 1 year of daily data
                    st.warning(f"Not enough data to process {ticker}." if language == 'English' else f"{ticker}的數據不足，無法處理。")
                    continue

                # Feature Engineering: Add Technical Indicators
                data_df = data.to_frame(name='Adj Close')
                data_df['Returns'] = data_df['Adj Close'].pct_change()
                data_df['MA5'] = data_df['Adj Close'].rolling(window=5).mean()
                data_df['MA10'] = data_df['Adj Close'].rolling(window=10).mean()
                data_df['MA20'] = data_df['Adj Close'].rolling(window=20).mean()
                data_df['STD5'] = data_df['Adj Close'].rolling(window=5).std()
                data_df['STD10'] = data_df['Adj Close'].rolling(window=10).std()
                data_df['STD20'] = data_df['Adj Close'].rolling(window=20).std()
                data_df = data_df.dropna()

                # Split features and target
                features = data_df.drop(columns=['Adj Close'])
                target = data_df['Adj Close']

                # Prepare exogenous variables
                if use_exogenous and exog_df_combined is not None:
                    exog = exog_df_combined.loc[target.index]
                    exog = exog.fillna(method='ffill').fillna(method='bfill')
                else:
                    exog = None

                # Implement Time Series Cross-Validation (Walk-Forward Validation)
                from sklearn.model_selection import TimeSeriesSplit

                tscv = TimeSeriesSplit(n_splits=5)
                models = ['ARIMA', 'Prophet', 'XGBoost', 'LSTM']
                model_performance = {model_name: {'MAE': [], 'RMSE': [], 'MAPE': []} for model_name in models}

                for train_index, test_index in tscv.split(target):
                    train_target, test_target = target.iloc[train_index], target.iloc[test_index]
                    train_features, test_features = features.iloc[train_index], features.iloc[test_index]
                    if exog is not None:
                        exog_train, exog_test = exog.iloc[train_index], exog.iloc[test_index]
                    else:
                        exog_train, exog_test = None, None

                    # ARIMA Model
                    with st.spinner('Running ARIMA model...' if language == 'English' else '正在運行ARIMA模型...'):
                        try:
                            model_autoARIMA = auto_arima(
                                train_target,
                                exogenous=exog_train,
                                start_p=1,
                                start_q=1,
                                max_p=max_p,
                                max_q=max_q,
                                max_d=max_d,
                                m=m if seasonal else 1,
                                seasonal=seasonal,
                                trace=False,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True,
                                information_criterion='aic',
                                max_order=20,
                                n_jobs=-1
                            )
                            order = model_autoARIMA.order
                            seasonal_order = model_autoARIMA.seasonal_order

                            model = SARIMAX(
                                train_target,
                                exog=exog_train,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            model_fit = model.fit(disp=False)

                            # Forecast
                            forecast = model_fit.forecast(steps=len(test_target), exog=exog_test)
                            # Evaluate
                            mae = mean_absolute_error(test_target, forecast)
                            rmse = np.sqrt(mean_squared_error(test_target, forecast))
                            mape = np.mean(np.abs((test_target - forecast) / test_target)) * 100

                            model_performance['ARIMA']['MAE'].append(mae)
                            model_performance['ARIMA']['RMSE'].append(rmse)
                            model_performance['ARIMA']['MAPE'].append(mape)
                        except Exception as e:
                            if debug_mode:
                                st.error(f"ARIMA failed for {ticker}: {e}")

                    # Prophet Model
                    with st.spinner('Running Prophet model...' if language == 'English' else '正在運行Prophet模型...'):
                        try:
                            prophet_train = pd.DataFrame({'ds': train_target.index, 'y': train_target.values})
                            prophet_test = pd.DataFrame({'ds': test_target.index, 'y': test_target.values})
                            if use_exogenous and exog_train is not None:
                                for col in exog_train.columns:
                                    prophet_train[col] = exog_train[col].values
                                    prophet_test[col] = exog_test[col].values
                                prophet_model = Prophet(daily_seasonality=seasonal)
                                for col in exog_train.columns:
                                    prophet_model.add_regressor(col)
                            else:
                                prophet_model = Prophet(daily_seasonality=seasonal)
                            prophet_model.fit(prophet_train)

                            forecast = prophet_model.predict(prophet_test)
                            forecast_values = forecast['yhat'].values

                            # Evaluate
                            mae = mean_absolute_error(test_target, forecast_values)
                            rmse = np.sqrt(mean_squared_error(test_target, forecast_values))
                            mape = np.mean(np.abs((test_target - forecast_values) / test_target)) * 100

                            model_performance['Prophet']['MAE'].append(mae)
                            model_performance['Prophet']['RMSE'].append(rmse)
                            model_performance['Prophet']['MAPE'].append(mape)
                        except Exception as e:
                            if debug_mode:
                                st.error(f"Prophet failed for {ticker}: {e}")

                    # XGBoost Model
                    with st.spinner('Running XGBoost model...' if language == 'English' else '正在運行XGBoost模型...'):
                        try:
                            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
                            xgb_model.fit(train_features, train_target)
                            forecast = xgb_model.predict(test_features)

                            # Evaluate
                            mae = mean_absolute_error(test_target, forecast)
                            rmse = np.sqrt(mean_squared_error(test_target, forecast))
                            mape = np.mean(np.abs((test_target - forecast) / test_target)) * 100

                            model_performance['XGBoost']['MAE'].append(mae)
                            model_performance['XGBoost']['RMSE'].append(rmse)
                            model_performance['XGBoost']['MAPE'].append(mape)
                        except Exception as e:
                            if debug_mode:
                                st.error(f"XGBoost failed for {ticker}: {e}")

                    # LSTM Model
                    with st.spinner('Running LSTM model...' if language == 'English' else '正在運行LSTM模型...'):
                        try:
                            from sklearn.preprocessing import MinMaxScaler

                            scaler = MinMaxScaler(feature_range=(0, 1))
                            scaled_train = scaler.fit_transform(train_target.values.reshape(-1, 1))

                            # Prepare data for LSTM
                            def create_dataset(dataset, look_back=1):
                                X, Y = [], []
                                for i in range(len(dataset) - look_back):
                                    X.append(dataset[i:(i + look_back), 0])
                                    Y.append(dataset[i + look_back, 0])
                                return np.array(X), np.array(Y)

                            look_back = 5
                            X_train_lstm, Y_train_lstm = create_dataset(scaled_train, look_back)
                            X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))

                            # Build LSTM Model
                            lstm_model = Sequential()
                            lstm_model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
                            lstm_model.add(Dense(1))
                            lstm_model.compile(loss='mean_squared_error', optimizer='adam')
                            lstm_model.fit(X_train_lstm, Y_train_lstm, epochs=10, batch_size=16, verbose=0)

                            # Prepare test data
                            total_data = np.concatenate((train_target.values, test_target.values))
                            inputs = total_data[len(total_data) - len(test_target) - look_back:]
                            inputs_scaled = scaler.transform(inputs.reshape(-1, 1))
                            X_test_lstm = []
                            for i in range(look_back, len(inputs_scaled)):
                                X_test_lstm.append(inputs_scaled[i - look_back:i, 0])
                            X_test_lstm = np.array(X_test_lstm)
                            X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

                            # Forecast
                            forecast_scaled = lstm_model.predict(X_test_lstm)
                            forecast = scaler.inverse_transform(forecast_scaled).flatten()

                            # Align forecast with test_target
                            test_target_lstm = test_target.values[:len(forecast)]

                            # Evaluate
                            mae = mean_absolute_error(test_target_lstm, forecast)
                            rmse = np.sqrt(mean_squared_error(test_target_lstm, forecast))
                            mape = np.mean(np.abs((test_target_lstm - forecast) / test_target_lstm)) * 100

                            model_performance['LSTM']['MAE'].append(mae)
                            model_performance['LSTM']['RMSE'].append(rmse)
                            model_performance['LSTM']['MAPE'].append(mape)
                        except Exception as e:
                            if debug_mode:
                                st.error(f"LSTM failed for {ticker}: {e}")

                # Aggregate Model Performance
                performance_summary = {}
                for model_name in models:
                    if len(model_performance[model_name]['MAE']) > 0:
                        mae_mean = np.mean(model_performance[model_name]['MAE'])
                        rmse_mean = np.mean(model_performance[model_name]['RMSE'])
                        mape_mean = np.mean(model_performance[model_name]['MAPE'])
                        performance_summary[model_name] = {'MAE': mae_mean, 'RMSE': rmse_mean, 'MAPE': mape_mean}
                    else:
                        performance_summary[model_name] = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

                # Store results
                results[ticker] = performance_summary

                # Save performance metrics to the database
                for model_name, metrics in performance_summary.items():
                    cursor.execute('''
                        INSERT INTO performance_metrics (ticker, model_name, mae, rmse, mape)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (ticker, model_name, metrics['MAE'], metrics['RMSE'], metrics['MAPE']))
                conn.commit()

                # Create tabs for each ticker with the new order
                with st.expander(f"{ticker} Analysis Details", expanded=False):
                    tab1, tab2, tab3 = st.tabs([
                        "Forecasting" if language == 'English' else "預測",
                        "Backtesting Performance" if language == 'English' else "回測表現",
                        "Historical Data" if language == 'English' else "歷史數據"
                    ])

                    with tab1:
                        st.subheader("Forecasting" if language == 'English' else "預測")
                        # Prepare future dates and features before forecasting
                        future_dates = pd.date_range(start=target.index[-1] + pd.DateOffset(days=1), periods=forecast_periods, freq='D')

                        # Prepare future exogenous variables
                        if use_exogenous and exog_df_combined is not None:
                            # For simplicity, we'll assume exogenous variables remain constant
                            last_exog_values = exog.iloc[-1].values.reshape(1, -1)
                            future_exog = np.repeat(last_exog_values, forecast_periods, axis=0)
                            future_exog = pd.DataFrame(future_exog, columns=exog.columns, index=future_dates)
                        else:
                            future_exog = None

                        # Initialize lists for successful forecasts and their weights
                        successful_forecasts = []
                        successful_weights = []
                        successful_model_names = []
                        model_forecasts = {}

                        # Collect MAPEs for weighting
                        model_mapes = {model: performance_summary[model]['MAPE'] for model in models}

                        # Find the best model based on MAPE
                        best_model_name = min(performance_summary, key=lambda k: performance_summary[k]['MAPE'])
                        best_model_mape = performance_summary[best_model_name]['MAPE']

                        # Forecast with all models
                        for model_name in models:
                            if model_name == 'ARIMA':
                                with st.spinner('Running ARIMA model...' if language == 'English' else '正在運行ARIMA模型...'):
                                    try:
                                        model_autoARIMA_full = auto_arima(
                                            target,
                                            exogenous=exog,
                                            start_p=1,
                                            start_q=1,
                                            max_p=max_p,
                                            max_q=max_q,
                                            max_d=max_d,
                                            m=m if seasonal else 1,
                                            seasonal=seasonal,
                                            trace=False,
                                            error_action='ignore',
                                            suppress_warnings=True,
                                            stepwise=True,
                                            information_criterion='aic',
                                            max_order=20,
                                            n_jobs=-1
                                        )
                                        order_full = model_autoARIMA_full.order
                                        seasonal_order_full = model_autoARIMA_full.seasonal_order

                                        model_full = SARIMAX(
                                            target,
                                            exog=exog,
                                            order=order_full,
                                            seasonal_order=seasonal_order_full,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False
                                        )
                                        model_full_fit = model_full.fit(disp=False)

                                        # Forecast
                                        forecast_arima = model_full_fit.forecast(steps=forecast_periods, exog=future_exog)
                                        successful_forecasts.append(forecast_arima.values)
                                        successful_model_names.append('ARIMA')
                                        model_forecasts['ARIMA'] = forecast_arima.values
                                    except Exception as e:
                                        if debug_mode:
                                            st.error(f"ARIMA forecasting failed for {ticker}: {e}")

                            elif model_name == 'Prophet':
                                with st.spinner('Running Prophet model...' if language == 'English' else '正在運行Prophet模型...'):
                                    try:
                                        prophet_full = pd.DataFrame({'ds': target.index, 'y': target.values})
                                        if use_exogenous and exog is not None:
                                            for col in exog.columns:
                                                prophet_full[col] = exog[col].values
                                            prophet_model = Prophet(daily_seasonality=seasonal)
                                            for col in exog.columns:
                                                prophet_model.add_regressor(col)
                                        else:
                                            prophet_model = Prophet(daily_seasonality=seasonal)
                                        prophet_model.fit(prophet_full)

                                        if use_exogenous and future_exog is not None:
                                            prophet_future = pd.DataFrame({'ds': future_dates})
                                            for col in future_exog.columns:
                                                prophet_future[col] = future_exog[col].values
                                        else:
                                            prophet_future = pd.DataFrame({'ds': future_dates})

                                        forecast = prophet_model.predict(prophet_future)
                                        forecast_prophet = forecast['yhat'].values
                                        successful_forecasts.append(forecast_prophet)
                                        successful_model_names.append('Prophet')
                                        model_forecasts['Prophet'] = forecast_prophet
                                    except Exception as e:
                                        if debug_mode:
                                            st.error(f"Prophet forecasting failed for {ticker}: {e}")

                            elif model_name == 'XGBoost':
                                with st.spinner('Running XGBoost model...' if language == 'English' else '正在運行XGBoost模型...'):
                                    try:
                                        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
                                        xgb_model.fit(features, target)
                                        # Prepare future features (technical indicators)
                                        last_values = features.iloc[-1]
                                        future_features = pd.DataFrame(index=future_dates)
                                        for col in features.columns:
                                            future_features[col] = last_values[col]
                                        forecast_xgb = xgb_model.predict(future_features)
                                        successful_forecasts.append(forecast_xgb)
                                        successful_model_names.append('XGBoost')
                                        model_forecasts['XGBoost'] = forecast_xgb
                                    except Exception as e:
                                        if debug_mode:
                                            st.error(f"XGBoost forecasting failed for {ticker}: {e}")

                            elif model_name == 'LSTM':
                                with st.spinner('Running LSTM model...' if language == 'English' else '正在運行LSTM模型...'):
                                    try:
                                        scaler = MinMaxScaler(feature_range=(0, 1))
                                        scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))

                                        # Prepare data for LSTM
                                        def create_dataset(dataset, look_back=1):
                                            X, Y = [], []
                                            for i in range(len(dataset) - look_back):
                                                X.append(dataset[i:(i + look_back), 0])
                                                Y.append(dataset[i + look_back, 0])
                                            return np.array(X), np.array(Y)

                                        look_back = 5
                                        X_full, Y_full = create_dataset(scaled_target, look_back)
                                        X_full = X_full.reshape((X_full.shape[0], X_full.shape[1], 1))

                                        # Build LSTM Model
                                        lstm_model = Sequential()
                                        lstm_model.add(LSTM(50, input_shape=(X_full.shape[1], X_full.shape[2])))
                                        lstm_model.add(Dense(1))
                                        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
                                        lstm_model.fit(X_full, Y_full, epochs=10, batch_size=16, verbose=0)

                                        # Forecast
                                        last_data = scaled_target[-look_back:].reshape(1, look_back, 1)
                                        forecast_lstm = []
                                        for _ in range(forecast_periods):
                                            pred = lstm_model.predict(last_data)
                                            forecast_lstm.append(pred[0][0])

                                            # Reshape pred to match dimensions
                                            pred_reshaped = pred.reshape((1, 1, 1))
                                            last_data = np.concatenate((last_data[:, 1:, :], pred_reshaped), axis=1)

                                        forecast_lstm = scaler.inverse_transform(np.array(forecast_lstm).reshape(-1, 1)).flatten()
                                        successful_forecasts.append(forecast_lstm)
                                        successful_model_names.append('LSTM')
                                        model_forecasts['LSTM'] = forecast_lstm
                                    except Exception as e:
                                        if debug_mode:
                                            st.error(f"LSTM forecasting failed for {ticker}: {e}")

                        # Collect MAPEs and weights for successful models
                        successful_mapes = [model_mapes[model] for model in successful_model_names if not np.isnan(model_mapes[model])]
                        total_inverse_mape = sum(1 / mape for mape in successful_mapes)
                        successful_weights = [(1 / model_mapes[model]) / total_inverse_mape for model in successful_model_names if not np.isnan(model_mapes[model])]

                        # Check if we have any successful forecasts
                        if successful_forecasts and successful_weights:
                            # Weighted Ensemble Forecast
                            forecasts_array = np.array([model_forecasts[model] for model in successful_model_names])
                            ensemble_forecast = np.average(forecasts_array, axis=0, weights=successful_weights)

                            predicted_prices_ensemble = pd.Series(ensemble_forecast, index=future_dates)

                            latest_price = latest_prices[ticker]
                            last_forecast_price_ensemble = predicted_prices_ensemble.iloc[-1]
                            ratio_ensemble = (last_forecast_price_ensemble / latest_price - 1) * 100  # Percentage change

                            # Display ensemble predicted price and comparison
                            st.metric(
                                label=f"Ensembled Predicted Price on {future_dates[-1].strftime('%Y-%m-%d')}",
                                value=f"${last_forecast_price_ensemble:.2f}",
                                delta=f"{ratio_ensemble:.2f}%"
                            )

                            # Best Model Forecast
                            if best_model_name in model_forecasts:
                                best_forecast_values = model_forecasts[best_model_name]
                                predicted_prices_best = pd.Series(best_forecast_values, index=future_dates)

                                last_forecast_price_best = predicted_prices_best.iloc[-1]
                                ratio_best = (last_forecast_price_best / latest_price - 1) * 100  # Percentage change

                                # Display best model predicted price and comparison
                                st.metric(
                                    label=f"Predicted Price on {future_dates[-1].strftime('%Y-%m-%d')} using {best_model_name}",
                                    value=f"${last_forecast_price_best:.2f}",
                                    delta=f"{ratio_best:.2f}%"
                                )

                                # Plot forecasts
                                fig_forecast = go.Figure()
                                fig_forecast.add_trace(go.Scatter(x=target.index, y=target.values, mode='lines', name='Historical'))
                                fig_forecast.add_trace(go.Scatter(x=predicted_prices_ensemble.index, y=predicted_prices_ensemble.values, mode='lines+markers', name='Ensemble Forecast'))
                                fig_forecast.add_trace(go.Scatter(x=predicted_prices_best.index, y=predicted_prices_best.values, mode='lines+markers', name=f'{best_model_name} Forecast'))
                                fig_forecast.update_layout(
                                    title=f'{ticker} Forecasted Prices',
                                    xaxis_title='Date',
                                    yaxis_title='Price'
                                )
                                st.plotly_chart(fig_forecast)

                                # Save forecast to the database
                                cursor.execute('''
                                    INSERT INTO forecasts (ticker, forecast_date, predicted_price, delta, model_names)
                                    VALUES (?, ?, ?, ?, ?)
                                ''', (
                                    ticker,
                                    future_dates[-1].strftime('%Y-%m-%d'),
                                    last_forecast_price_ensemble,
                                    ratio_ensemble,
                                    ', '.join(successful_model_names)
                                ))
                                conn.commit()
                            else:
                                st.warning(f"Best model ({best_model_name}) did not produce a forecast." if language == 'English' else f"最佳模型（{best_model_name}）未能生成預測。")
                        else:
                            st.warning(f"No successful forecasts were generated for {ticker}." if language == 'English' else f"未能為 {ticker} 生成成功的預測。")

                    with tab2:
                        st.subheader("Backtesting Performance" if language == 'English' else "回測表現")
                        performance_df = pd.DataFrame(performance_summary).T
                        st.dataframe(performance_df.style.format("{:.2f}"))

                        # Plot model performance comparison
                        fig_perf = go.Figure()
                        fig_perf.add_trace(go.Bar(
                            x=performance_df.index,
                            y=performance_df['MAPE'],
                            text=performance_df['MAPE'].apply(lambda x: f"{x:.2f}%"),
                            textposition='auto'
                        ))
                        fig_perf.update_layout(
                            title='Model Performance Comparison (MAPE)' if language == 'English' else '模型性能比較（MAPE）',
                            xaxis_title='Model' if language == 'English' else '模型',
                            yaxis_title='MAPE (%)'
                        )
                        st.plotly_chart(fig_perf)

                        # Display best model
                        st.write(f"Best Model: **{best_model_name}**" if language == 'English' else f"最佳模型：**{best_model_name}**")

                    with tab3:
                        st.subheader("Historical Data" if language == 'English' else "歷史數據")
                        if debug_mode:
                            st.dataframe(data_df.reset_index())

                        # Plot historical data
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Scatter(x=data_df.index, y=data_df['Adj Close'], mode='lines', name='Adjusted Close'))
                        fig_hist.update_layout(
                            title=f'{ticker} Historical Prices' if language == 'English' else f'{ticker} 歷史價格',
                            xaxis_title='Date' if language == 'English' else '日期',
                            yaxis_title='Price' if language == 'English' else '價格'
                        )
                        st.plotly_chart(fig_hist)

                end_time_ticker = time.time()
                elapsed_time_ticker = end_time_ticker - start_time_ticker
                if debug_mode:
                    st.write(f"Time taken to process {ticker}: {elapsed_time_ticker:.2f} seconds")

                # Update progress bar
                progress_bar.progress((idx + 1) * progress_step)

            if results:
                st.markdown("## Model Evaluation Metrics" if language == 'English' else "## 模型評估指標")
                for ticker in results:
                    st.subheader(f"{ticker}")
                    performance_df = pd.DataFrame(results[ticker]).T
                    st.dataframe(performance_df.style.format("{:.2f}"))
            else:
                st.warning("No results to display." if language == 'English' else "沒有結果可顯示。")

            # Close the database connection after analysis
            conn.close()

    else:
        st.info("Configure the parameters above and click **Start Analysis** to begin." if language == 'English' else "請配置以上參數，然後點擊 **開始分析** 以開始。")

elif app_mode == 'View Saved Results' if language == 'English' else '查看已保存的結果':
    st.header('Saved Results' if language == 'English' else '已保存的結果')

    # Connect to the database
    conn = sqlite3.connect('files/Ticker_results.db')
    cursor = conn.cursor()

    # Get list of tickers from the database
    cursor.execute('SELECT DISTINCT ticker FROM forecasts')
    tickers = [row[0] for row in cursor.fetchall()]

    if tickers:
        selected_ticker = st.selectbox('Select Ticker to View Results' if language == 'English' else '選擇要查看結果的標的', tickers)

        # Display performance metrics
        st.subheader('Performance Metrics' if language == 'English' else '性能指標')
        cursor.execute('''
            SELECT model_name, mae, rmse, mape, date
            FROM performance_metrics
            WHERE ticker = ?
            ORDER BY date DESC
        ''', (selected_ticker,))
        metrics = cursor.fetchall()
        metrics_df = pd.DataFrame(metrics, columns=['Model', 'MAE', 'RMSE', 'MAPE', 'Date'] if language == 'English' else ['模型', 'MAE', 'RMSE', 'MAPE', '日期'])
        st.dataframe(metrics_df)

        # Display forecasts
        st.subheader('Forecasts' if language == 'English' else '預測')
        cursor.execute('''
            SELECT forecast_date, predicted_price, delta, model_names, date
            FROM forecasts
            WHERE ticker = ?
            ORDER BY date DESC
        ''', (selected_ticker,))
        forecasts = cursor.fetchall()
        forecasts_df = pd.DataFrame(forecasts, columns=['Forecast Date', 'Predicted Price', 'Delta (%)', 'Models Used', 'Date'] if language == 'English' else ['預測日期', '預測價格', '變化 (%)', '使用的模型', '日期'])
        st.dataframe(forecasts_df)
    else:
        st.info("No saved results found." if language == 'English' else "未找到已保存的結果。")

    # Close the database connection
    conn.close()