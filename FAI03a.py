'''
#1. 下載 20250630~20250930 市值前 100 之上市股票之調整後收盤價，並估計各檔股票的
IRR、年化 IRR、年化報酬率的 sigma、Sharpe ratio (=年化 IRR /年化報酬率的
sigma)，並將上述 4 種股權資產的參數輸出到 FAI03a_股票數據分析.xlsx，並輸出下列問
題的答案: (1)以年化 IRR 為投資績效評估，則哪一檔上市股票的績效最好? (2)以 Sharpe
ratio 為投資績效評估，則哪一檔上市股票的績效最好?
#2. 下載 20250630~20250930 市值前 50 之上櫃股票之調整後收盤價，並估計各檔股票的
IRR、年化 IRR、年化報酬率的 sigma、Sharpe ratio (=年化 IRR /年化報酬率的
sigma)，並將上述 4 種股權資產的參數輸出到 FAI03b_股票數據分析.xlsx，並輸出下列
問題的答案: (1)以年化 IRR 為投資績效評估，則哪一檔上櫃股票的績效最好? (2)以
Sharpe ratio 為投資績效評估，則哪一檔上櫃股票的績效最好?
#3. 下載 20250630~20250930 ETF: [0050, 0056, 00878, 00919, 006208, 00980A, 00981A,
00982A, 00983A, 00984A] 之調整後收盤價，並估計各檔股票的 IRR、年化 IRR、年化報
酬率的 sigma、Sharpe ratio (=年化 IRR /年化報酬率的 sigma)，並將上述 4 種股權資產
的參數輸出到 FAI03c_股票數據分析.xlsx，並輸出下列問題的答案: (1)以年化 IRR 為投
資績效評估，則哪一檔上櫃股票的績效最好? (2)以 Sharpe ratio 為投資績效評估，則哪
一檔上櫃股票的績效最好?
'''
import yfinance as yf
import pandas as pd
import numpy as np
import math

yr_day= 250                   
TW_PickStockNo=100           
start3 = '2025-06-30'
end3 = '2025-10-01'

TW_stock_ticker =  pd.read_html('https://www.taifex.com.tw/cht/9/futuresQADetail' , encoding='utf-8')[0]['證券名稱'][:TW_PickStockNo]
TW_stock_list = [str(i)+'.TW' for i in TW_stock_ticker]
TW_100 = yf.download(TW_stock_list, start3, end3)['Close']

R_TW_100 = np.log(TW_100).diff()
STD_TW_100_annual = np.std(R_TW_100)*(math.sqrt(yr_day))

N_days = R_TW_100.shape[0]
IRR = (TW_100.iloc[-1] / TW_100.iloc[0]) - 1
Annual_IRR = (TW_100.iloc[-1] / TW_100.iloc[0]) ** (yr_day / N_days) - 1
SR_TW_100 = Annual_IRR / STD_TW_100_annual

col2 = "IRR"
col3 = "Annual_IRR"
col4 = "STD"
col5 = "Sharpe ratio"

TW_100_vs = pd.DataFrame({col2:IRR, col3:Annual_IRR, col4:STD_TW_100_annual, col5:SR_TW_100})
TW_100_vs.to_excel('FAI03a_股票數據分析.xlsx', index=True)

TW_100_list = list(SR_TW_100.index)
SR_TW_100_max = np.argmax(SR_TW_100)
TW_100_best = TW_100_list[SR_TW_100_max]
print('\n', '以 Sharpe ratio 績效最佳:', TW_100_best)

TW_100_list = list(Annual_IRR.index)
Annual_IRR_TW_100_max = np.argmax(Annual_IRR)
TW_100_best = TW_100_list[Annual_IRR_TW_100_max]
print('\n', '以 年化IRR 績效最佳:', TW_100_best)