'''
請上網抓取 20221118 ~ 20251118，台灣上市前 100 檔股票的收盤價，計算各檔股價
的報酬率，並以 JB test 檢定每一檔股票的報酬率是否服從常態分配，如下列表格，
將結果輸出到 FAI07b.xlsx。
'''

import yfinance as yf
from scipy import stats
import pandas as pd
import numpy as np

start = '2022-11-18'
end = '2025-11-19'
stock_ticker = pd.read_html('https://www.taifex.com.tw/cht/9/futuresQADetail' , encoding='utf-8')[0]['證券名稱'][:100]
ticker_list = [str(i)+'.TW' for i in stock_ticker]
TW_data = yf.download(ticker_list, start, end)['Close']
alpha = 0.01

r = np.log(TW_data/TW_data.shift()).dropna(how='any')
result_table = pd.DataFrame(index = ticker_list, columns = ['mean', 'std','skew', 'kurt', 'p_value', 'normality'])
for i in result_table.index:
    R = r[i]
    result_table.loc[i, 'mean'] = R.mean()
    result_table.loc[i,'std'] = R.std(ddof = 1)
    result_table.loc[i, 'skew'] = R.skew()
    result_table.loc[i, 'kurt'] = R.kurt()
    
    jb_stat, p_value = stats.jarque_bera(R)
    result_table.loc[i, 'p_value'] = p_value
    
    if p_value > alpha:
        result_table.loc[i, 'normality'] = True
    else:
        result_table.loc[i, 'normality'] = False
        
result_table.to_excel('FAI07b_股票數據分析.xlsx')    