import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
from pmdarima import auto_arima
import warnings
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(layout="centered", page_title="ETF Price Forecasting with ARIMA Models")

# Streamlit app title
st.title("📈 ETF Price Forecasting with ARIMA, SARIMA, and ARIMAX Models")

# Instructions
st.markdown("""
This app performs time series forecasting on selected ETFs using ARIMA, SARIMA, and ARIMAX models.
It displays backtesting results, predicted prices, and interactive plots.
""")

# Language selection
language = st.selectbox('Choose Language / 選擇語言', ['English', '繁體中文'])

# Debug mode
debug_mode = st.checkbox("Show Debug Information" if language == 'English' else "顯示除錯信息", value=False)

# List of ETFs
etfs = ['RSP', 'QQQ', 'MOAT', 'PFF', 'VNQ', 'IWY']

# User Inputs
st.header("Input Parameters" if language == 'English' else "輸入參數")

# ETF selection
selected_etfs = st.multiselect(
    'Select ETFs to Analyze' if language == 'English' else '選擇要分析的ETF',
    etfs,
    default=etfs  # Default to all ETFs
)

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
    # Forecast periods
    forecast_periods = st.slider(
        'Forecast Periods (Months Ahead)' if language == 'English' else '預測期（未來幾個月）',
        min_value=1,
        max_value=12,
        value=1,
        step=1
    )
    # Number of months for rolling forecast evaluation
    rolling_months = st.slider(
        'Number of Months for Backtesting' if language == 'English' else '回測月份數',
        min_value=6,
        max_value=24,
        value=12,
        step=1
    )

with st.expander("Model Parameters" if language == 'English' else "模型參數"):
    colA, colB = st.columns(2)
    
    with colA:
        # Model parameters with explanations
        st.markdown("**ARIMA Model Parameters**")
        max_p = st.slider('Max p (AR order)' if language == 'English' else '最大p值（AR階數）',
                        0, 5, 2, help='Maximum number of lag observations included in the model (AutoRegressive terms).')
        max_d = st.slider('Max d (Differencing order)' if language == 'English' else '最大d值（差分階數）',
                        0, 2, 1, help='Number of times the data is differenced to make it stationary.')
        max_q = st.slider('Max q (MA order)' if language == 'English' else '最大q值（MA階數）',
                        0, 5, 2, help='Maximum size of the moving average window (Moving Average terms).')

    with colB:
        st.markdown("**Seasonal Parameters**")
        seasonal = st.checkbox('Include Seasonality (SARIMA)' if language == 'English' else '包含季節性（SARIMA）', value=False)
        m = st.number_input('Seasonal Period (m)' if language == 'English' else '季節性週期（m）',
                            min_value=1, value=12, step=1, help='Number of time steps for a single seasonal period.')

        # Exogenous variables (for ARIMAX)
        st.markdown("**Include Exogenous Variables (ARIMAX)**")
        use_exogenous = st.checkbox('Use Exogenous Variables' if language == 'English' else '使用外生變數', value=False)
        if use_exogenous:
            exogenous_options = ['^GSPC', '^IXIC', '^DJI']  # S&P 500, NASDAQ, Dow Jones indices
            selected_exogenous = st.multiselect(
                'Select Exogenous Variables' if language == 'English' else '選擇外生變數',
                exogenous_options,
                default=['^GSPC']
            )
        else:
            selected_exogenous = []
    # About Section
with st.expander("About the Models and Metrics" if language == 'English' else "關於模型和指標"):
    st.markdown("""
    **Time Series Forecasting Models**:

    1. **ARIMA (AutoRegressive Integrated Moving Average)**:
    - Combines autoregression (AR), differencing (I), and moving average (MA) components.
    - Suitable for non-seasonal data.

    2. **SARIMA (Seasonal ARIMA)**:
    - Extends ARIMA by adding seasonal components.
    - Captures repeating patterns over fixed periods.

    3. **ARIMAX (ARIMA with Exogenous Variables)**:
    - Incorporates external variables that may influence the target variable.
    - Useful when external factors impact the time series.

    **Model Parameters**:

    - **p**: Number of autoregressive terms.
    - **d**: Number of non-seasonal differences needed for stationarity.
    - **q**: Number of lagged forecast errors in the prediction equation.
    - **P**: Number of seasonal autoregressive terms.
    - **D**: Number of seasonal differences.
    - **Q**: Number of seasonal moving average terms.
    - **m**: The number of time steps for a single seasonal period.

    **Error Metrics**:

    - **MAE (Mean Absolute Error)**:
    - Measures the average magnitude of errors in a set of forecasts.
    - Less sensitive to outliers compared to RMSE.

    - **RMSE (Root Mean Squared Error)**:
    - Measures the square root of the average of squared differences between forecasted and observed values.
    - Gives higher weight to large errors.

    - **MAPE (Mean Absolute Percentage Error)**:
    - Expresses accuracy as a percentage.
    - Scale-independent, making it easier to interpret.

    **Additional Resources**:

    - [Introduction to ARIMA Models](https://www.statsmodels.org/stable/examples/notebooks/generated/arima.html)
    - [Time Series Analysis and Forecasting](https://otexts.com/fpp2/)
    - [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)

    """)
# Check if start date is before end date
if start_date >= end_date:
    st.error("Error: End date must be after start date." if language == 'English' else "錯誤：結束日期必須在開始日期之後。")
    st.stop()

# Button to start analysis
if st.button('Start Analysis' if language == 'English' else '開始分析'):
    st.markdown(f"## {'Analysis Results' if language == 'English' else '分析結果'}")

    # Initialize dictionaries
    etf_data = {}
    latest_prices = {}
    results = {}

    # Progress bar
    progress_bar = st.progress(0)
    progress_step = 1 / len(selected_etfs) if selected_etfs else 1

    # Fetch exogenous data if needed
    if use_exogenous and selected_exogenous:
        st.write("Fetching exogenous data..." if language == 'English' else "正在獲取外生變數數據...")
        exog_data = {}
        for exog_symbol in selected_exogenous:
            exog_df = yf.download(exog_symbol, start=start_date, end=end_date + pd.DateOffset(days=1), progress=False)
            exog_df = exog_df['Adj Close'].resample('M').last()
            exog_df = exog_df[exog_df.index <= current_date]
            exog_data[exog_symbol] = exog_df
        exog_df_combined = pd.DataFrame(exog_data)
        exog_df_combined = exog_df_combined.fillna(method='ffill').fillna(method='bfill')
    else:
        exog_df_combined = None

    # Process each ETF
    for idx, etf in enumerate(selected_etfs):
        st.markdown(f"### {etf}")
        start_time_etf = time.time()
        data = yf.download(etf, start=start_date, end=end_date + pd.DateOffset(days=1), progress=False)

        # Check if data is empty
        if data.empty:
            st.warning(f"No data found for {etf}. Skipping..." if language == 'English' else f"未找到{etf}的數據，跳過...")
            continue

        # Resample to monthly data (last available date of each month up to current date)
        data = data['Adj Close'].resample('M').last()
        # Filter out future dates beyond the current date
        data = data[data.index <= current_date]
        etf_data[etf] = data
        # Store the latest closing price
        latest_prices[etf] = data.iloc[-1]

        # Ensure the data has enough points
        if len(data) < 36:  # Minimum 3 years of monthly data
            st.warning(f"Not enough data to process {etf}." if language == 'English' else f"{etf}的數據不足，無法處理。")
            continue

        # Align exogenous data with ETF data
        if use_exogenous and exog_df_combined is not None:
            exog = exog_df_combined.loc[data.index]
        else:
            exog = None

        # Initialize lists to store forecasts and actuals
        forecasts = []
        actuals = []
        dates = []

        # Total number of data points
        N = len(data)

        # Rolling Forecast Evaluation over the past N months
        for i in range(N - rolling_months, N):
            train_data = data[:i]
            test_data = data[i:i+1]

            # Exogenous data for training and testing
            if exog is not None:
                exog_train = exog.iloc[:i]
                exog_test = exog.iloc[i:i+1]
            else:
                exog_train = None
                exog_test = None

            # Check if test_data is not empty
            if test_data.empty:
                if debug_mode:
                    st.warning(f"No test data available for {etf} at index {i}.")
                continue

            # Check if train_data is sufficient
            if len(train_data) < 24:  # Minimum 2 years of training data
                if debug_mode:
                    st.warning(f"Not enough training data for {etf} at index {i}.")
                continue

            # Use auto_arima with user-specified parameters
            try:
                model_autoARIMA = auto_arima(
                    train_data,
                    exogenous=exog_train,
                    start_p=0,
                    start_q=0,
                    max_p=max_p,
                    max_q=max_q,
                    max_d=max_d,
                    m=m if seasonal else 1,
                    seasonal=seasonal,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    information_criterion='aic'
                )
                order = model_autoARIMA.order
                seasonal_order = model_autoARIMA.seasonal_order
            except Exception as e:
                if debug_mode:
                    st.error(f"auto_arima failed for {etf} at index {i}: {e}")
                continue

            # Fit the SARIMAX model
            try:
                model = SARIMAX(
                    train_data,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_fit = model.fit(disp=False)
                # Forecast
                forecast = model_fit.forecast(steps=1, exog=exog_test)
                # Append to lists
                forecasts.append(forecast.iloc[0])
                actuals.append(test_data.iloc[0])
                dates.append(test_data.index[0])
            except Exception as e:
                if debug_mode:
                    st.error(f"Model fitting failed for {etf} at index {i}: {e}")
                continue

        # Step 2: Evaluate Model
        if len(forecasts) > 0 and len(actuals) > 0 and len(dates) > 0:
            forecasts = pd.Series(forecasts, index=dates)
            actuals = pd.Series(actuals, index=dates)

            mae = mean_absolute_error(actuals, forecasts)
            rmse = np.sqrt(mean_squared_error(actuals, forecasts))
            mape = np.mean(np.abs((actuals - forecasts) / actuals)) * 100

            # Store results
            results[etf] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

            # Create tabs for each ETF
            with st.expander(f"{etf} Analysis Details" if language == 'English' else f"{etf} 分析詳情", expanded=False):
                tab1, tab2, tab3 = st.tabs([
                    "Data Overview" if language == 'English' else "數據概覽",
                    "Backtesting Results" if language == 'English' else "回測結果",
                    "Forecasting" if language == 'English' else "預測"                
                    ])

                with tab1:
                    st.subheader("Historical Data" if language == 'English' else "歷史數據")
                    if debug_mode:
                        st.dataframe(data.reset_index())

                    # Plot historical data
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Adjusted Close'))
                    fig_hist.update_layout(
                        title=f'{etf} Historical Prices' if language == 'English' else f'{etf} 歷史價格',
                        xaxis_title='Date' if language == 'English' else '日期',
                        yaxis_title='Price' if language == 'English' else '價格'
                    )
                    st.plotly_chart(fig_hist)

                with tab2:
                    st.subheader("Backtesting Performance" if language == 'English' else "回測表現")

                    # Display metrics with explanations
                    col1_bt, col2_bt, col3_bt = st.columns(3)
                    col1_bt.metric("MAE", f"{mae:.2f}", help="Mean Absolute Error: Average absolute difference between predicted and actual values.")
                    col2_bt.metric("RMSE", f"{rmse:.2f}", help="Root Mean Squared Error: Square root of the average squared differences between predicted and actual values.")
                    col3_bt.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error: Average absolute percentage difference between predicted and actual values.")

                    # Assess Accuracy
                    if mape < 5:
                        accuracy = "highly accurate" if language == 'English' else "高度準確"
                    elif mape < 10:
                        accuracy = "moderately accurate" if language == 'English' else "中度準確"
                    else:
                        accuracy = "not very accurate" if language == 'English' else "不太準確"

                    st.write(f"The model's backtesting indicates it is **{accuracy}** in forecasting {etf} prices." if language == 'English' else f"模型的回測表明其在預測 {etf} 價格方面 **{accuracy}**。")

                    # Plot backtesting results
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=actuals.index, y=actuals, mode='lines+markers', name='Actual'))
                    fig_bt.add_trace(go.Scatter(x=forecasts.index, y=forecasts, mode='lines+markers', name='Forecast'))
                    fig_bt.update_layout(
                        title=f'{etf} Backtesting Results' if language == 'English' else f'{etf} 回測結果',
                        xaxis_title='Date' if language == 'English' else '日期',
                        yaxis_title='Price' if language == 'English' else '價格'
                    )
                    st.plotly_chart(fig_bt)

                with tab3:
                    st.subheader("Forecasting" if language == 'English' else "預測")

                    try:
                        # Use the entire data to fit the model
                        if exog is not None:
                            exog_full = exog
                        else:
                            exog_full = None

                        model_autoARIMA_full = auto_arima(
                            data,
                            exogenous=exog_full,
                            start_p=0,
                            start_q=0,
                            max_p=max_p,
                            max_q=max_q,
                            max_d=max_d,
                            m=m if seasonal else 1,
                            seasonal=seasonal,
                            trace=False,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True,
                            information_criterion='aic'
                        )
                        order_full = model_autoARIMA_full.order
                        seasonal_order_full = model_autoARIMA_full.seasonal_order

                        model_full = SARIMAX(
                            data,
                            exog=exog_full,
                            order=order_full,
                            seasonal_order=seasonal_order_full,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        model_full_fit = model_full.fit(disp=False)

                        # Prepare exogenous variables for forecasting
                        if exog is not None:
                            # Assume exogenous variables remain constant for forecasting
                            last_exog = exog.iloc[-1].values.reshape(1, -1)
                            exog_future = np.repeat(last_exog, forecast_periods, axis=0)
                        else:
                            exog_future = None

                        # Forecast the next periods
                        forecast = model_full_fit.forecast(steps=forecast_periods, exog=exog_future)
                        forecast_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='M')
                        predicted_prices = pd.Series(forecast.values, index=forecast_dates)

                        latest_price = latest_prices[etf]
                        last_forecast_price = predicted_prices.iloc[-1]
                        ratio = (last_forecast_price / latest_price - 1) * 100  # Percentage change

                        # Display predicted price and comparison
                        st.metric(
                            label=f"Predicted Price on {forecast_dates[-1].strftime('%Y-%m-%d')}",
                            value=f"${last_forecast_price:.2f}",
                            delta=f"{ratio:.2f}%"
                        )

                        st.write(f"Latest Closing Price (as of {data.index[-1].strftime('%Y-%m-%d')}): **${latest_price:.2f}**" if language == 'English' else f"最近收盤價（截至 {data.index[-1].strftime('%Y-%m-%d')}）：**${latest_price:.2f}**")

                        # Plot forecast
                        fig_forecast = go.Figure()
                        fig_forecast.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Historical'))
                        fig_forecast.add_trace(go.Scatter(x=predicted_prices.index, y=predicted_prices, mode='lines+markers', name='Forecast'))
                        fig_forecast.update_layout(
                            title=f'{etf} Forecasted Prices' if language == 'English' else f'{etf} 預測價格',
                            xaxis_title='Date' if language == 'English' else '日期',
                            yaxis_title='Price' if language == 'English' else '價格'
                        )
                        st.plotly_chart(fig_forecast)

                    except Exception as e:
                        st.error(f"Failed to predict future prices for {etf}: {e}" if language == 'English' else f"無法預測 {etf} 的未來價格：{e}")

        else:
            st.warning(f"Not enough forecasts to evaluate the model for {etf}." if language == 'English' else f"沒有足夠的預測來評估 {etf} 的模型。")

        end_time_etf = time.time()
        elapsed_time_etf = end_time_etf - start_time_etf
        if debug_mode:
            st.write(f"Time taken to process {etf}: {elapsed_time_etf:.2f} seconds")

        # Update progress bar
        progress_bar.progress((idx + 1) * progress_step)

    if results:
        st.markdown("## Model Evaluation Metrics" if language == 'English' else "## 模型評估指標")
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df.style.format("{:.2f}"))
    else:
        st.warning("No results to display." if language == 'English' else "沒有結果可顯示。")

else:
    st.info("Configure the parameters above and click **Start Analysis** to begin." if language == 'English' else "請配置以上參數，然後點擊 **開始分析** 以開始。")

