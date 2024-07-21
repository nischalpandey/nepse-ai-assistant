import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
import os
import io

import base64

def generate_da():
    json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'latestdata/priceNEPSE.json')

    # Load the json data from the file
    stocks_data = pd.read_json(json_file)
    
    # Data preprocessing
    stocks_data = stocks_data.drop(['s'], axis=1)
    stocks_data = stocks_data.set_index('t')
    stocks_data.index = pd.to_datetime(stocks_data.index, unit='s')

    html_plots = "The following are the visualizations of the stock data: <br> <br>"
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f'<img src="data:image/png;base64,{img_str}"/>'


    # Close Price and Volume plot
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    stocks_data['c'].plot(ax=ax[0])
    stocks_data['v'].plot(ax=ax[1])
    ax[0].set_title('Close Price and Volume')
    ax[0].set_ylabel('Close Price')
    ax[1].set_ylabel('Volume in Arab')
    plt.tight_layout()
    html_plots += fig_to_base64(fig)

    # Daily Return plot
    stocks_data['daily_return'] = stocks_data['c'].pct_change()
    fig, ax = plt.subplots(figsize=(15, 7))
    stocks_data['daily_return'].plot(ax=ax)
    ax.set_title('Daily Return')
    ax.set_ylabel('Percentage Change')
    html_plots += fig_to_base64(fig)

    # 30 Day Volatility plot
    fig, ax = plt.subplots(figsize=(15, 7))
    stocks_data['c'].rolling(window=30).std().plot(ax=ax)
    ax.set_title('30 Day Volatility')
    ax.set_ylabel('Standard Deviation')
    html_plots += fig_to_base64(fig)

    # Moving Averages plot
    ma_day = [10, 20, 50]
    for ma in ma_day:
        column_name = f"MA for {ma} days"
        stocks_data[column_name] = stocks_data['c'].rolling(ma).mean()
    fig, ax = plt.subplots(figsize=(15, 7))
    stocks_data[['c', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=ax)
    ax.set_title('Moving Average')
    html_plots += fig_to_base64(fig)

    # 30 Day Volatility plot again (it seems repeated)
    fig, ax = plt.subplots(figsize=(15, 7))
    stocks_data['c'].rolling(window=30).std().plot(ax=ax)
    ax.set_title('30 Day Volatility')
    ax.set_ylabel('Standard Deviation')
    html_plots += fig_to_base64(fig)

    # Correlation heatmap
    stocks_data = stocks_data.drop(['daily_return', 'volatility'], axis=1, errors='ignore')
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(stocks_data.corr(), annot=True, ax=ax)
    html_plots += fig_to_base64(fig)

    return html_plots




