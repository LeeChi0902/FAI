

import yfinance as yf
import pandas as pd
import numpy as np
import math

yr_day= 250                   
TW_PickStockNo= 50           
start3 = '2025-06-30'
end3 = '2025-10-01'

TW_stock_ticker =  pd.read_html('https://www.taifex.com.tw/cht/2/tPEXPropertion' , encoding='utf-8')[0]['證券名稱'][:TW_PickStockNo]
TW_stock_list = [str(i)+'.TWO' for i in TW_stock_ticker]
TW_50 = yf.download(TW_stock_list, start3, end3)['Close']

R_TW_50 = np.log(TW_50).diff()
STD_TW_50_annual = np.std(R_TW_50)*(math.sqrt(yr_day))

N_days = R_TW_50.shape[0]
IRR = (TW_50.iloc[-1] / TW_50.iloc[0]) - 1
Annual_IRR = (TW_50.iloc[-1] / TW_50.iloc[0]) ** (yr_day / N_days) - 1
SR_TW_50 = Annual_IRR / STD_TW_50_annual

col2 = "IRR"
col3 = "Annual_IRR"
col4 = "STD"
col5 = "Sharpe ratio"

TW_50_vs = pd.DataFrame({col2:IRR, col3:Annual_IRR, col4:STD_TW_50_annual, col5:SR_TW_50})
TW_50_vs.to_excel('FAI03b_股票數據分析.xlsx', index=True)

TW_50_list = list(SR_TW_50.index)
SR_TW_50_max = np.argmax(SR_TW_50)
TW_50_best = TW_50_list[SR_TW_50_max]
print('\n', '以 Sharpe ratio 績效最佳:', TW_50_best)

TW_50_list = list(Annual_IRR.index)
Annual_IRR_TW_50_max = np.argmax(Annual_IRR)
TW_50_best = TW_50_list[Annual_IRR_TW_50_max]
print('\n', '以 年化IRR 績效最佳:', TW_50_best)