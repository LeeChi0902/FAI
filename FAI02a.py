'''
#1. 下載 20250623~20250923 市值前 100 之上市股票之調整後收盤價，並估計各檔股票的年
化報酬率的均值、年化報酬率的 sigma、Sharpe ratio，並將上述 3 種股權資產的參數輸
出到 FAI02a_股票數據分析.xlsx，並輸出下列問題的答案: (1)以年化報酬率為投資績效評
估，則哪一檔上市股票的績效最好? (2)以 Sharpe ratio 為投資績效評估，則哪一檔上市
股票的績效最好?
#2. 下載 20250623~20250923 市值前 50 之上櫃股票之調整後收盤價，並估計各檔股票的年化
報酬率的均值、年化報酬率的 sigma、Sharpe ratio，並將上述 3 種股權資產的參數輸出
到 FAI02b_股票數據分析.xlsx，並輸出下列問題的答案: (1)以年化報酬率為投資績效評
估，則哪一檔上櫃股票的績效最好? (2)以 Sharpe ratio 為投資績效評估，則哪一檔上櫃
股票的績效最好?
'''
import yfinance as yf
import pandas as pd
import numpy as np
import math

yr_day=248                    
TW_PickStockNo=100           

start3 = '2025-06-23'
end3 = '2025-09-24'


TW_stock_ticker =  pd.read_html('https://www.taifex.com.tw/cht/9/futuresQADetail' , encoding='utf-8')[0]['證券名稱'][:TW_PickStockNo]

TW_100 = pd.DataFrame()
R_TW_100 = pd.DataFrame()
TW_stock_list = []


for i in TW_stock_ticker:
    TW_stock_list.append(str(i)+'.TW')

TW_100 = yf.download(TW_stock_list, start3, end3)
TW_100 = TW_100['Close']
R_TW_100 = np.log(TW_100).diff()
R_TW_100_annual = R_TW_100.mean()*yr_day
STD_TW_100_annual = np.std(R_TW_100)*(math.sqrt(yr_day))
SR_TW_100 = R_TW_100_annual/STD_TW_100_annual

col2 = "Mean"
col3 = "STD"
col4 = "Sharpe ratio"

TW_100_vs = pd.DataFrame({col2:R_TW_100_annual, col3:STD_TW_100_annual, col4:SR_TW_100})
TW_100_vs.to_excel('FAI02a_股票數據分析.xlsx', index=True)

TW_100_list = list(SR_TW_100.index)
SR_TW_100_max = np.argmax(SR_TW_100)
TW_100_best = TW_100_list[SR_TW_100_max]
print('\n', '以 Sharpe ratio 績效最佳:', TW_100_best)

TW_100_list = list(R_TW_100_annual.index)
Mean_TW_100_max = np.argmax(R_TW_100_annual)
TW_100_best = TW_100_list[Mean_TW_100_max]
print('\n', '以 年化報酬率 績效最佳:', TW_100_best)

