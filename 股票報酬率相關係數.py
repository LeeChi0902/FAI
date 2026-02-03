#股票相關係數
import pandas as pd
import numpy as np
import yfinance as yf

stock_number = 50
s_d = pd.read_html('https://www.taifex.com.tw/cht/9/futuresQADetail')[0]

print(s_d.columns)


s_d = s_d[['排行', '證券名稱']]
s_d = s_d.iloc[0:stock_number]

s_d['證券名稱'] = [str(i) + '.TW' for i in s_d['證券名稱']]
df = yf.download(s_d['證券名稱'].tolist(), start = '2022-04-01', end = '2023-03-31')
df = df["Close"]
daily_r = (df-df.shift(1))/df.shift(1)
corr_matrix = daily_r.corr()
I = np.zeros_like(corr_matrix)
I[np.tril_indices_from(I)] = True
corr_matrix_lower = corr_matrix*I
corr_matrix_lower[corr_matrix_lower == 0] = np.nan
corr_matrix_lower.to_excel('0704_stock_correlation.xlsx')