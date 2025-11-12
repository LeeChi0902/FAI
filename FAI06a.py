'''
請上網抓取 20250821~20251021 台灣上市股票市值前 100 檔股票的收盤價，計算此期
間每天的股票日報酬率，完成下列工作:
1. 以市場模型(Market model) 搭配 OLS 計算每一檔股票的 α 與 β，另外再計算每一檔
股票此期間的 IRR。將每一檔股票的 IRR、α、β 依據股票的代碼，如下列表格所
示，輸出到 FAI06.xlsx。
2. 繪畫出：(1)上面 α (X 軸) 與相對應之 IRR(Y 軸)的二維圖形，(2)上面 β (X 軸) 與相對
應之 IRR(Y 軸)的二維圖形。
'''
#1.做三個function可以單獨計算單檔股票的Alpha, Beta, IRR
#2.用一個DataFrame搭配for迴圈去接各檔股票數值
#3.再把DataFrame裡面部不同Columns抓出來當X軸Y軸

import yfinance as yf
import numpy as np
import matplotlib.pylab as plt
import statsmodels.api as sm
import pandas as pd


def Alpha(data:pd.Series, market_data:pd.Series):
    Y = np.log(data/data.shift())
    X = np.log(market_data/market_data.shift())
    df = pd.concat([Y, X], axis = 1).dropna()
    Y = df.iloc[:, 0]
    X = sm.add_constant(df.iloc[:, 1])
    result = sm.OLS(Y, X).fit()
    return result.params[0]

def Beta(data:pd.Series, market_data:pd.Series):
    Y = np.log(data/data.shift())
    X = np.log(market_data/market_data.shift())
    df = pd.concat([Y, X], axis = 1).dropna()
    Y = df.iloc[:, 0]
    X = sm.add_constant(df.iloc[:, 1])
    result = sm.OLS(Y, X).fit()
    return result.params[1]

def IRR(data:pd.Series):
    s = data.dropna()
    start = s.iloc[0]
    end = s.iloc[-1]
    IRR  = end/start -1
    return IRR


TW_stock_ticker =  pd.read_html('https://www.taifex.com.tw/cht/9/futuresQADetail' , encoding='utf-8')[0]['證券名稱'][:100]
TW_stock_list = [str(i)+'.TW' for i in TW_stock_ticker]
market_ticker = '^TWII'
start = '2025-08-21'
end = '2025-10-22'
TW_data = yf.download(TW_stock_list, start, end)['Close']
market_data = yf.download(market_ticker, start, end)['Close']

result_table = pd.DataFrame(index = TW_stock_list, columns = ['Alpha', 'Beta', 'IRR'])
for stock in result_table.index:
    result_table.loc[stock, 'Alpha'] = Alpha(TW_data[stock], market_data)
    result_table.loc[stock, 'Beta'] = Beta(TW_data[stock], market_data)
    result_table.loc[stock, 'IRR'] = IRR(TW_data[stock])

result_table.to_excel('FAI06a_股票數據分析.xlsx')

plt.plot(result_table['Alpha'], result_table['IRR'], 'bo', markersize=3)
plt.xlabel('Alpha')
plt.ylabel('IRR')
plt.show()
plt.plot(result_table['Beta'], result_table['IRR'], 'bo', markersize=3)
plt.xlabel('Beta')
plt.ylabel('IRR')
plt.show()
