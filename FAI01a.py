#

import yfinance as yf
import pandas as pd
import numpy as np
import math

yr_day=248   
start2 = '2025-06-16'
end2 = '2025-09-17'

etf_quotation1 = ['0050.TW']
etf_quotation2 = ['0051.TW']
etf_quotation3 = ['0056.TW']
etf_quotation4 = ['00981A.TW']
etf_quotation5 = ['00982A.TW']
etf_quotation6 = ['00980A.TW']

df7 = pd.DataFrame()
df7 = yf.download(etf_quotation1, start=start2, end=end2)
df8 = pd.DataFrame()
df8 = yf.download(etf_quotation2, start=start2, end=end2)
df9 = pd.DataFrame()
df9 = yf.download(etf_quotation3, start=start2, end=end2)
df10 = pd.DataFrame()
df10 = yf.download(etf_quotation4, start=start2, end=end2)
df11 = pd.DataFrame()
df11 = yf.download(etf_quotation5, start=start2, end=end2)
df12 = pd.DataFrame()
df12 = yf.download(etf_quotation6, start=start2, end=end2)


df7['Return'] = np.log(df7['Close']).diff()
R_annual_mean7 = np.mean(df7['Return'])*yr_day
R_annual_std7 = np.std(df7['Return'])*(math.sqrt(yr_day))
SR7 = R_annual_mean7/R_annual_std7

df8['Return'] = np.log(df8['Close']).diff()
R_annual_mean8 = np.mean(df8['Return'])*yr_day
R_annual_std8 = np.std(df8['Return'])*(math.sqrt(yr_day))
SR8 = R_annual_mean8/R_annual_std8

df9['Return'] = np.log(df9['Close']).diff()
R_annual_mean9 = np.mean(df9['Return'])*yr_day
R_annual_std9 = np.std(df9['Return'])*(math.sqrt(yr_day))
SR9 = R_annual_mean9/R_annual_std9

df10['Return'] = np.log(df10['Close']).diff()
R_annual_mean10 = np.mean(df10['Return'])*yr_day
R_annual_std10 = np.std(df10['Return'])*(math.sqrt(yr_day))
SR10 = R_annual_mean10/R_annual_std10

df11['Return'] = np.log(df11['Close']).diff()
R_annual_mean11 = np.mean(df11['Return'])*yr_day
R_annual_std11 = np.std(df11['Return'])*(math.sqrt(yr_day))
SR11 = R_annual_mean11/R_annual_std11

df12['Return'] = np.log(df12['Close']).diff()
R_annual_mean12 = np.mean(df12['Return'])*yr_day
R_annual_std12 = np.std(df12['Return'])*(math.sqrt(yr_day))
SR12 = R_annual_mean12/R_annual_std12

with pd.ExcelWriter("FAI 第02章 PPT02_01b ETF數據aReturn.xlsx") as writer2:
    df7.to_excel(writer2, sheet_name="0050")  
    df8.to_excel(writer2, sheet_name="0051")  
    df9.to_excel(writer2, sheet_name="0056") 
    df10.to_excel(writer2, sheet_name="00981A") 
    df11.to_excel(writer2, sheet_name="00982A") 
    df12.to_excel(writer2, sheet_name="00980A")  
  

TWETF_list = ['0050','0051','0056','00981A','00982A','00980A']
list_mean2 = [R_annual_mean7, R_annual_mean8, R_annual_mean9, R_annual_mean10, R_annual_mean11, R_annual_mean12]
list_std2 = [R_annual_std7, R_annual_std8, R_annual_std9, R_annual_std10, R_annual_std11, R_annual_std12]
list_SR2 = [SR7, SR8, SR9, SR10, SR11, SR12]
col1 = "Stock"
col2 = "Mean"
col3 = "std"
col4 = "Sharpe ratio"

df_etf = pd.DataFrame({col1:TWETF_list, col2:list_mean2, col3:list_std2, col4: list_SR2})
df_etf.to_excel('FAI 第02章 PPT02_01b ETF數據b績效.xlsx', index=False)

list_SR2_max = np.argmax(list_SR2)
TWETF_best = TWETF_list[list_SR2_max]
print("\n","TWETF Sharpe ratio最高，績效最好:",TWETF_best)

