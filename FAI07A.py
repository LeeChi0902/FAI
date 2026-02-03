'''
請上網抓取 20221118 ~ 20251118，台灣上市前 100 檔股票的收盤價，根據 30 日滾動
視窗的方式計算各檔股票之股價報酬率的前四個動差: {μ、σ、Skew、Kurt}，輸出
到 FAI07a.xlsx 的各個子工作表。
'''

import yfinance as yf
import numpy as np
import pandas as pd

def rolling_mean(data, run_day):
    data = data.dropna(how='any')
    R = np.log(data/data.shift())
    R_rolling_mean = R.rolling(window=run_day).mean().dropna(how='all')
    return R_rolling_mean

def rolling_std(data, run_day):
    data = data.dropna(how='any')
    R = np.log(data/data.shift())
    R_rolling_std = R.rolling(window=run_day).std().dropna(how='all')
    return R_rolling_std

def rolling_skew(data, run_day):
    data = data.dropna(how='any')
    R = np.log(data/data.shift())
    R_rolling_skew = R.rolling(window=run_day).skew().dropna(how='all')
    return R_rolling_skew

def rolling_kurt(data, run_day):
    data = data.dropna(how='any')
    R = np.log(data/data.shift())
    R_rolling_kurt = R.rolling(window=run_day).kurt().dropna(how='all')
    return R_rolling_kurt

start = '2022-11-18'
end = '2025-11-19'
stock_ticker = pd.read_html('https://www.taifex.com.tw/cht/9/futuresQADetail' , encoding='utf-8')[0]['證券名稱'][:100]
ticker_list = [str(i)+'.TW' for i in stock_ticker]
TW_data = yf.download(ticker_list, start, end)['Close']

with pd.ExcelWriter('FAI07a_股票數據分析.xlsx') as writer:
    rolling_mean(TW_data, 30).to_excel(writer, sheet_name = 'mean')
    rolling_std(TW_data, 30).to_excel(writer, sheet_name = 'std')
    rolling_skew(TW_data, 30).to_excel(writer, sheet_name = 'skew')
    rolling_kurt(TW_data, 30).to_excel(writer, sheet_name = 'kurt')