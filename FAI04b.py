import yfinance as yf
import pandas as pd
import numpy as np
import math

yr_day= 248                   
TW_PickStockNo=10           
start = '2025-09-23'
end = '2025-10-08'

TW_stock_ticker =  pd.read_html('https://www.taifex.com.tw/cht/9/futuresQADetail' , encoding='utf-8')[0]['證券名稱'][:TW_PickStockNo]
TWO_stock_ticker =  pd.read_html('https://www.taifex.com.tw/cht/2/tPEXPropertion' , encoding='utf-8')[0]['證券名稱'][:TW_PickStockNo]
TW_stock_list = [str(i)+'.TW' for i in TW_stock_ticker]
TWO_stock_list = [str(i)+'.TWO' for i in TWO_stock_ticker]
tickers = [i for i in TW_stock_list + TWO_stock_list + ['0050.TW']]
total_close = yf.download(tickers, start, end)['Close']

R_total = np.log(total_close).diff()
STD_total_annual = np.std(R_total)*(math.sqrt(yr_day))

IRR_list = []
Annual_IRR_list = []

for col in total_close.columns:
    s = total_close[col].dropna()
    start_p = s.iloc[0]     
    end_p   = s.iloc[-1]    
    irr = end_p / start_p - 1

    n_days = s.shape[0]      
    ann_irr = (end_p / start_p) ** (yr_day / n_days) - 1

    IRR_list.append(irr)
    Annual_IRR_list.append(ann_irr)

IRR = pd.Series(IRR_list, index = total_close.columns)
Annual_IRR = pd.Series(Annual_IRR_list, index = total_close.columns)
SR_total = Annual_IRR / STD_total_annual

col2 = "IRR"
col3 = "Annual_IRR"
col4 = "STD"
col5 = "Sharpe ratio"
total_vs = pd.DataFrame({col2:IRR, col3:Annual_IRR, col4:STD_total_annual, col5:SR_total})
'''
total_vs.to_excel('FAI04b_股票數據分析.xlsx', sheet_name='Total', index=True)
'''
irr_sorted = IRR.sort_values(ascending=False)
tw50_ticker = '0050.TW'

tw50_irr = irr_sorted.loc[tw50_ticker]
gt_tw50 = irr_sorted[irr_sorted > tw50_irr]

best_stock = irr_sorted.index[0]
best_irr = irr_sorted.iloc[0]

with pd.ExcelWriter('FAI04b_股票數據分析.xlsx', engine='openpyxl') as writer:
    gt_tw50.to_frame(name='IRR').to_excel(writer, sheet_name='Sheet1')

    pd.DataFrame({'股票代碼': [best_stock],'最高IRR': [best_irr]}).to_excel(writer, sheet_name='Sheet2', index=False)
    
    total_vs.to_excel(writer, sheet_name='Total', index=True)
    

