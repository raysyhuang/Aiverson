import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA


@st.cache_data
def get_stock_data(ticker, start_date, end_date, retries=3, delay=5):
    try:
        yq_ticker = yf.Ticker(ticker)
        yq_data = yq_ticker.history(start=start_date, end=end_date)

        if yq_data.empty:
            st.error(f"No data available for {ticker} using yfinance.")
            return pd.DataFrame()

        yq_data.reset_index(inplace=True)

        yq_data.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }, inplace=True)

        yq_data['Date'] = pd.to_datetime(yq_data['Date'], errors='coerce')
        yq_data['Date'] = yq_data['Date'].dt.tz_localize(None)

        return yq_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def calculate_metrics(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    data = data.dropna()
    return data


def plot_stock_data(data, ticker, language):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price' if language == 'English' else '收盤價'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], name='50-day SMA' if language == 'English' else '50日均線'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_200'], name='200-day SMA' if language == 'English' else '200日均線'))
    fig.update_layout(title=f'{ticker} Stock Price' if language == 'English' else f'{ticker} 股票價格', 
                      xaxis_title='Date' if language == 'English' else '日期', 
                      yaxis_title='Price' if language == 'English' else '價格')
    return fig


def predict_stock_price_arima(data, days_ahead=30, backtest_period='12 months', language='English', debug=False):
    """
    Predict future stock prices using ARIMA and perform backtesting.
    """
    try:
        df = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])

        # Ensure no future dates in historical data
        today = pd.Timestamp(datetime.now().date())
        df = df[df['ds'] <= today]

        # Backtesting
        months = int(backtest_period.split()[0])
        cutoff_date = df['ds'].max() - pd.DateOffset(months=months)
        train_df = df[df['ds'] <= cutoff_date]
        test_df = df[df['ds'] > cutoff_date]

        if len(train_df) < 2 or len(test_df) < 1:
            st.warning("Not enough data for backtesting." if language == 'English' else "沒有足夠的數據進行回測。")
            backtest_rmse = 0
            forecast_bt = pd.DataFrame()  # Empty DataFrame for backtesting forecast
        else:
            # Train ARIMA model on training data
            model_bt = ARIMA(train_df['y'], order=(5, 1, 0))
            model_bt_fit = model_bt.fit()

            # Forecast for backtesting period
            forecast_bt = model_bt_fit.forecast(steps=len(test_df))
            backtest_rmse = np.sqrt(mean_squared_error(test_df['y'], forecast_bt))
            if debug:
                st.write(f"**Backtesting RMSE:** {backtest_rmse:.2f}")

            # Convert forecast_bt into a DataFrame for plotting
            forecast_bt = pd.DataFrame({'ds': test_df['ds'], 'yhat': forecast_bt})

        # Future Forecasting
        last_date = df['ds'].max()
        model_full = ARIMA(df['y'], order=(5, 1, 0))
        model_full_fit = model_full.fit()

        future_forecast = model_full_fit.forecast(steps=days_ahead)
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead, freq='B')

        future_forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': future_forecast})

        if future_forecast_df.empty and debug:
            st.warning("Future forecast is empty. Check if 'days_ahead' is appropriate." if language == 'English' else "未來預測為空。檢查預測天數是否合適。")
        elif debug:
            st.write("**Future Forecast:**" if language == 'English' else "**未來預測:**")
            st.write(future_forecast_df)

        return future_forecast_df, forecast_bt, backtest_rmse, test_df
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}" if language == 'English' else f"預測時發生錯誤：{e}")
        return pd.DataFrame(), pd.DataFrame(), 0, pd.DataFrame()


def stock_tracker(ticker='AAPL', prediction_days=30, start_date=None, end_date=None):
    st.set_page_config(layout="centered", page_title="Enhanced Stock Tracker")
    st.title('Enhanced Stock Tracker and Analysis')

    # Language selection
    language = st.selectbox('Choose Language / 選擇語言', ['English', '繁體中文'])

    # Add a checkbox for debug mode
    debug_mode = st.checkbox("Show Debug Information" if language == 'English' else "顯示除錯信息", value=False)

    st.header("Input Parameters" if language == 'English' else "輸入參數")
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, GOOGL)' if language == 'English' else '輸入股票代碼（例如 AAPL、GOOGL）', ticker).upper()
    with col2:
        prediction_days = st.slider('Prediction Days' if language == 'English' else '預測天數', 7, 90, prediction_days)

    st.subheader("Select Date Range" if language == 'English' else "選擇日期範圍")
    today = datetime.today().date()
    default_start_date = today - timedelta(days=365 * 3)
    start_date = st.date_input('Start Date' if language == 'English' else '開始日期', default_start_date if start_date is None else start_date, max_value=today)
    end_date = st.date_input('End Date' if language == 'English' else '結束日期', today if end_date is None else end_date, min_value=start_date, max_value=today)

    if start_date > end_date:
        st.error("Error: Start Date must be before End Date." if language == 'English' else "錯誤：開始日期必須在結束日期之前。")
        return

    with st.spinner('Fetching and analyzing stock data...' if language == 'English' else '正在獲取和分析股票數據...'):
        start_date_dt = datetime.combine(start_date, datetime.min.time())
        end_date_dt = datetime.combine(end_date, datetime.min.time())
        data = get_stock_data(ticker, start_date_dt, end_date_dt)

    if not data.empty:
        required_columns = {'Date', 'Close'}
        if not required_columns.issubset(data.columns):
            st.error(f"Data is missing required columns: {required_columns - set(data.columns)}")
            return

        if debug_mode:
            st.write("**Data Retrieved Successfully:**" if language == 'English' else "**數據檢索成功：**")
            st.write(data.head())
            st.write(data.tail())
            st.write("**Data Date Range:**" if language == 'English' else "**數據日期範圍：**", data['Date'].min(), "to", data['Date'].max())

        today = pd.Timestamp(datetime.now().date())
        data = data[data['Date'] <= today]

        data = calculate_metrics(data)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f'{ticker} Stock Price Chart' if language == 'English' else f'{ticker} 股票價格圖')
            fig = plot_stock_data(data, ticker, language)
            st.plotly_chart(fig)

        with col2:
            st.subheader('Key Metrics' if language == 'English' else '關鍵指標')
            st.metric("Current Price" if language == 'English' else "當前價格", f"${data['Close'].iloc[-1]:.2f}",
                      f"{data['Daily_Return'].iloc[-1]:.2%}")
            st.metric("50-day SMA" if language == 'English' else "50日均線", f"${data['SMA_50'].iloc[-1]:.2f}")
            st.metric("200-day SMA" if language == 'English' else "200日均線", f"${data['SMA_200'].iloc[-1]:.2f}")
            st.metric("Volatility (Annualized)" if language == 'English' else "波動率（年化）", f"{data['Volatility'].iloc[-1]:.2%}")

        if debug_mode:
            st.subheader('Recent Stock Data' if language == 'English' else '最近的股票數據')
            st.dataframe(data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'Daily_Return']].tail())

        st.header('Advanced Analysis' if language == 'English' else '高級分析')
        tab1, tab2 = st.tabs(["Performance Metrics" if language == 'English' else "績效指標", "Price Prediction" if language == 'English' else "價格預測"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Performance Metrics' if language == 'English' else '績效指標')
                total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                st.write(f"Total Return: {total_return:.2f}%" if language == 'English' else f"總回報：{total_return:.2f}%")
                st.write(f"Average Daily Return: {data['Daily_Return'].mean():.2%}" if language == 'English' else f"平均日回報：{data['Daily_Return'].mean():.2%}")
                st.write(f"Max Drawdown: {(data['Close'] / data['Close'].cummax() - 1).min():.2%}" if language == 'English' else f"最大回撤：{(data['Close'] / data['Close'].cummax() - 1).min():.2%}")

            with col2:
                st.subheader('Risk Metrics' if language == 'English' else '風險指標')
                st.write(f"Annualized Volatility: {data['Volatility'].mean():.2%}" if language == 'English' else f"年化波動率：{data['Volatility'].mean():.2%}")
                sharpe_ratio = (data['Daily_Return'].mean() / data['Daily_Return'].std()) * np.sqrt(252)
                st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}" if language == 'English' else f"夏普比率：{sharpe_ratio:.2f}")

        with tab2:
            st.subheader(f'Price Prediction (Next {prediction_days} Days)' if language == 'English' else f'價格預測（接下來 {prediction_days} 天）')
            # Define the backtest period
            backtest_period = '12 months'

            future_forecast_df, forecast_bt, backtest_rmse, test_df = predict_stock_price_arima(data, days_ahead=prediction_days, backtest_period=backtest_period, language=language, debug=debug_mode)

            if not future_forecast_df.empty:
                st.write(f"Predicted price after {prediction_days} days:" if language == 'English' else f"預測價格在 {prediction_days} 天後：")
                if debug_mode:
                    st.write(future_forecast_df)

            st.write("""
            **Explanation of Backtesting RMSE:**
            The Backtesting RMSE (Root Mean Square Error) is a measure of how accurate the model's predictions were 
            compared to actual stock prices during the backtesting period (the last 12 months in this case). 
            A lower RMSE indicates that the model was more accurate, whereas a higher RMSE means there was 
            more error in the model's predictions.
            """ if language == 'English' else """
            **回測RMSE解釋：**
            回測RMSE（均方根誤差）是衡量模型預測與實際股價相比的準確性的一個指標
            在這種情況下是過去12個月的回測期
            RMSE越低，表明模型越準確，而RMSE越高，則意味著模型的預測誤差越大。
            """)

            if backtest_rmse > 0:
                st.write(f"Backtesting RMSE (last {backtest_period}): ${backtest_rmse:.2f}" if language == 'English' else f"回測RMSE（過去 {backtest_period}）：${backtest_rmse:.2f}")

            # Plotting actual vs forecasted data
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual Close Price' if language == 'English' else '實際收盤價'))

            if not forecast_bt.empty:
                fig.add_trace(go.Scatter(x=forecast_bt['ds'], y=forecast_bt['yhat'], name='Backtest Forecast' if language == 'English' else '回測預測', mode='lines'))

            if not future_forecast_df.empty:
                fig.add_trace(go.Scatter(x=future_forecast_df['ds'], y=future_forecast_df['yhat'], name='Future Forecast' if language == 'English' else '未來預測', mode='lines'))

            fig.update_layout(title=f'{ticker} Price Prediction and Backtesting' if language == 'English' else f'{ticker} 價格預測與回測', 
                              xaxis_title='Date' if language == 'English' else '日期', 
                              yaxis_title='Price' if language == 'English' else '價格')
            st.plotly_chart(fig)

    else:
        st.error("No data available for the selected stock and date range." if language == 'English' else "沒有可用的數據，請檢查選擇的股票和日期範圍。")


if __name__ == "__main__":
    stock_tracker()