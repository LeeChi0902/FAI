import yfinance as yf
import pandas as pd
import numpy as np
import math

yr_day = 248
TW_PickStockNo = 50   
start3 = '2025-06-23'
end3 = '2025-09-24'



df = pd.read_html('https://www.taifex.com.tw/cht/2/tPEXPropertion', header=0, encoding='utf-8')[0]

# 欄位處理（避免多層表頭或空白）
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [''.join(map(str, col)).strip() for col in df.columns]
else:
    df.columns = df.columns.str.strip()

# 自動尋找股票代號欄位（4碼數字）
candidate = None
for c in df.columns:
    s = df[c].astype(str).str.strip()
    if (s.str.match(r'^\d{4}$').sum() >= 10):
        candidate = c
        break
if candidate is None:
    raise KeyError(f"在欄位中找不到股票代號欄：{list(df.columns)}")

TW_stock_ticker = df[candidate].astype(str).str.extract(r'(\d{4})')[0].dropna().head(TW_PickStockNo)


TW_stock_list = [str(i)+'.TWO' for i in TW_stock_ticker]  

TW_50 = yf.download(TW_stock_list, start3, end3)['Close']

R_TW_50 = np.log(TW_50).diff()

R_TW_50_annual = R_TW_50.mean() * yr_day

STD_TW_50_annual = R_TW_50.std() * math.sqrt(yr_day)

SR_TW_50 = R_TW_50_annual / STD_TW_50_annual

col2 = "Mean"
col3 = "STD"
col4 = "Sharpe ratio"

TW_50_vs = pd.DataFrame({col2:R_TW_50_annual, col3:STD_TW_50_annual, col4:SR_TW_50})

TW_50_vs.to_excel('FAI02b_股票數據分析.xlsx', index=True)

TW_50_list = list(R_TW_50_annual.index)
Mean_TW_50_max = np.argmax(R_TW_50_annual)
TW_50_best_ret = TW_50_list[Mean_TW_50_max]

TW_50_list = list(SR_TW_50.index)
SR_TW_50_max = np.argmax(SR_TW_50)
TW_50_best_sr = TW_50_list[SR_TW_50_max]

print('\n(1) 以 年化報酬率 為投資績效評估，最佳股票:', TW_50_best_ret)
print('(2) 以 Sharpe ratio 為投資績效評估，最佳股票:', TW_50_best_sr)
