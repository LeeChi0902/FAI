import yfinance as yf
import pandas as pd
import numpy as np
import math

yr_day= 250                      
start3 = '2025-06-30'
end3 = '2025-10-01'

base_etfs = ["0050", "0056", "00878", "00919", "006208", "00980A", "00981A", "00982A", "00983A", "00984A"]
ETF_tickers = [t + ".TW" for t in base_etfs]
ETF = yf.download(ETF_tickers, start3, end3)['Close']

R_ETF = np.log(ETF).diff()
STD_ETF_annual = np.std(R_ETF)*(math.sqrt(yr_day))

#針對0094A2025/07/14才掛牌上市交易導致IRR相關參數無法與其他ETF計算做以下處理 把每一檔ETF分開來取iloc[0]跟iloc[1]的位置
IRR_list = []
Annual_IRR_list = []

for col in ETF.columns:
    s = ETF[col].dropna()
    if s.shape[0] < 2:   
        IRR_list.append(np.nan)
        Annual_IRR_list.append(np.nan)
        continue

    start_p = s.iloc[0]     
    end_p   = s.iloc[-1]    
    irr = end_p / start_p - 1

    n_days = s.shape[0]      
    ann_irr = (end_p / start_p) ** (yr_day / n_days) - 1

    IRR_list.append(irr)
    Annual_IRR_list.append(ann_irr)

IRR = pd.Series(IRR_list, index=ETF.columns)
Annual_IRR = pd.Series(Annual_IRR_list, index=ETF.columns)

SR_ETF = Annual_IRR / STD_ETF_annual

col2 = "IRR"
col3 = "Annual_IRR"
col4 = "STD"
col5 = "Sharpe ratio"

ETF_vs = pd.DataFrame({col2:IRR, col3:Annual_IRR, col4:STD_ETF_annual, col5:SR_ETF})
ETF_vs.to_excel('FAI03c_股票數據分析.xlsx', index=True)

ETF_list = list(SR_ETF.index)
SR_ETF_max = np.argmax(SR_ETF)
ETF_best = ETF_list[SR_ETF_max]
print('\n', '以 Sharpe ratio 績效最佳:', ETF_best)

ETF_list = list(Annual_IRR.index)
Annual_IRR_ETF_max = np.argmax(Annual_IRR)
ETF_best = ETF_list[Annual_IRR_ETF_max]
print('\n', '以 年化IRR 績效最佳:', ETF_best)