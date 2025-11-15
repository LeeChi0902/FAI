'''
請上網抓取台灣上市前 100 檔股票以及上櫃前 50 檔股票的股票代碼存放在 list，再以迴
圈搭配 yfinance 下載 150 檔股票調整後收盤價，完成下列工作: (Sharpe ratio
=IRRy/σ，IRRy=年化內部報酬率、σ=年化報酬率波動度)
1. 請問上市股票市值前 100 檔的近 3 天 IRR 最高 10 檔股票(由大到小排序)為何?請將此
10 檔股票近 3 天、5 天、10 天、20 天、60 天、120 天、240 天、480 天、720 天 IRR
輸出到: FAI05a_股票數據分析.xlsx
2. 請問上市股票市值前 100 檔的近 10 天 Sharpe ratio 最高 10 檔股票(由大到小排序)為
何? 請將此 10 檔股票近 3 天、5 天、10 天、20 天、60 天、120 天、240 天、480
天、720 天 Sharpe ratio 輸出到: FAI05b_股票數據分析.xlsx
3. 請問上櫃股票市值前 50 檔的近 3 天 IRR 最高 10 檔股票(由大到小排序)為何?請將此
10 檔股票近 3 天、5 天、10 天、20 天、60 天、120 天、240 天、480 天、720 天 IRR
輸出到: FAI05c_股票數據分析.xlsx
4. 請問上櫃股票市值前 50 檔的近 10 天 Sharpe ratio 最高 10 檔股票(由大到小排序)為
何?請將此 10 檔股票近 3 天、5 天、10 天、20 天、60 天、120 天、240 天、480
天、720 天 Sharpe ratio 輸出到: FAI05d_股票數據分析.xlsx
5. 請問上市前 100+上櫃前 50 股票近 3 天 IRR 最高 10 檔股票(由大到小排序)為何?請將
此 10 檔股票近 3 天、5 天、10 天、20 天、60 天、120 天、240 天、480 天、720 天
IRR 輸出到: FAI05e_股票數據分析.xlsx
6. 請問上市前 100+上櫃前 50 股票近 10 天 Sharpe ratio 最高 10 檔股票(由大到小排序)為
何?請將此 10 檔股票近 3 天、5 天、10 天、20 天、60 天、120 天、240 天、480
天、720 天 Sharpe ratio 輸出到: FAI05f_股票數據分析.xlsx
註: IRR=累積內部報酬率、IRRy=年化內部報酬率。3 天無法算出 σ，因此 Sharpe ratio
改為 10 天
'''
#1.先下載近三年資料
#2.做一個比較近x天IRR並列出前y名的funtion
#3.做一個比較近x天shapre並列出前y名的function
#4.做一個計算單一標的物近x天IRR的functoin
#5.做一個計算單一標的物近x天sharpe的funtion

import yfinance as yf
import math
import numpy as np
import pandas as pd


def top_IRR_series(inputdata:pd.DataFrame, day:int, rank:int):
    start = inputdata.iloc[-1-day]
    end = inputdata.iloc[-1]
    irr = end/start -1
    irr_sorted = irr.sort_values(ascending = False)
    top = irr_sorted.head(rank)
    return top

def top_Sharpe_series(inputdata:pd.DataFrame, day:int, rank:int):
    start = inputdata.iloc[-1-day]
    end = inputdata.iloc[-1]
    annual_IRR = (end/start)**(248/day) -1
    
    SubData = inputdata.iloc[-1-day:]
    R_SubData = np.log(SubData).diff()
    annual_Std_SubData = (R_SubData.std(axis = 0, ddof = 1))*(math.sqrt(248))
    
    SR_SubData = annual_IRR/annual_Std_SubData
    SR_sorted = SR_SubData.sort_values(ascending = False)
    top = SR_sorted.head(rank)
    return top

def n_days_IRR(stock_name:str, day:int):
    start = Total_data[stock_name].iloc[-1-day]
    end = Total_data[stock_name].iloc[-1]
    irr = end/start -1
    return irr

def n_days_SR(stock_name:str, day:int):
    start = Total_data[stock_name].iloc[-1-day]
    end = Total_data[stock_name].iloc[-1]
    annual_irr = (end/start)**(248/day) -1
    
    SubData = Total_data[stock_name].iloc[-1-day:]
    R_SubData = np.log(SubData).diff()
    annual_Std_SubData = (R_SubData.std(ddof = 1))*(math.sqrt(248))
    
    SR_SubData = annual_irr/annual_Std_SubData
    return SR_SubData

def process(data:pd.DataFrame, label_1:str, label_2:str):
    top10irr_3d = top_IRR_series(data, 3, 10)
    result_top10_IRR = pd.DataFrame(index = top10irr_3d.index, columns = [f'IRR_{d}d' for d in days_list])
    for stock in result_top10_IRR.index:
        for d in days_list:
            result_top10_IRR.loc[stock,f'IRR_{d}d'] = n_days_IRR(stock, d)
    result_top10_IRR.to_excel(f'FAI05{label_1}_股票數據分析.xlsx')

    top10sr_10d = top_Sharpe_series(data, 10, 10)
    result_top10_SR = pd.DataFrame(index = top10sr_10d.index, columns = [f'SR_{d}d' for d in days_list])
    for stock in result_top10_SR.index:
        for d in days_list:
            result_top10_SR.loc[stock,f'SR_{d}d'] = n_days_SR(stock, d)
    result_top10_SR.to_excel(f'FAI05{label_2}_股票數據分析.xlsx')


TW_stock_ticker =  pd.read_html('https://www.taifex.com.tw/cht/9/futuresQADetail' , encoding='utf-8')[0]['證券名稱'][:100]
TWO_stock_ticker =  pd.read_html('https://www.taifex.com.tw/cht/2/tPEXPropertion' , encoding='utf-8')[0]['證券名稱'][:50]
TWO_stock_list = [str(i)+'.TWO' for i in TWO_stock_ticker]
TW_stock_list = [str(i)+'.TW' for i in TW_stock_ticker]
#給標籤,方便後面單獨計算
market_map = {}
for i in TW_stock_list:
    market_map[i] = 'TW'
for i in TWO_stock_list:
    market_map[i] = 'TWO'
Total_list = TW_stock_list + TWO_stock_list
Total_data = yf.download(Total_list, period='3y')['Close']
TW_data = Total_data[[i for i in Total_data.columns if market_map[i] == 'TW']]
TWO_data = Total_data[[i for i in Total_data.columns if market_map[i] == 'TWO']]
days_list = [3, 5, 10, 20, 60, 120, 240, 480, 720]

process(TW_data, 'a', 'b')
process(TWO_data, 'c', 'd')
process(Total_data, 'e', 'f')

