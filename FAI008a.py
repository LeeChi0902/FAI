'''
1. 請上網抓取 20250710~20250910 (第 t 期)，台灣大盤、台灣上市前 100 檔股票的調整
後收盤價，計算股價報酬率估計每一檔股票的 {α、β、Skew、Kurt、IRR}，每一檔
股票會有五組參數，請輸出到 FAI08a.xlsx，如下列表格所示，
以 20250910~20251110 (第 t+1 期)估計IRRt +1，以上述 20250710~20250910 估計的自變數
進行 OLS 檢定
IRRt =a0+ a1 αt + a2 βt +a3 Skewt + a4 Kurtt +ε t (1)
IRRt +1=a0 +a1 αt +a2 βt +a3 Skewt +a4 Kurt t+ε t (2)
測試完成請將兩 OLS 公式的 4 個回歸參數之係數{a1 、 a2 、 a3 、 a4}以及參數係數的 p-
value 分別輸出到 FAI08b.xlsx 。
'''
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

def R(data:pd.Series):
    R = np.log(data/data.shift()).dropna()
    return R

def IRR(data):
    IRR = (data.iloc[-1]/data.iloc[0])-1
    return IRR

def ols(target_series, factor_df):
    df = pd.concat([target_series, factor_df], axis = 1).dropna(how='any')
    df = df.astype(float)
    Y = df.iloc[:,0]
    X = sm.add_constant(df.iloc[:,1:])
    result = sm.OLS(Y,X).fit()
    return result

t0_start='2025-07-10'
t0_end='2025-09-11'
t1_start='2025-09-10'
t1_end='2025-11-11'
stock_ticker = pd.read_html('https://www.taifex.com.tw/cht/9/futuresQADetail' , encoding='utf-8')[0]['證券名稱'][:100]
ticker_list = [str(i)+'.TW' for i in stock_ticker]
data_t0 = yf.download(ticker_list, t0_start, t0_end)['Close']
data_t1 = yf.download(ticker_list, t1_start, t1_end)['Close']
data_market = yf.download('^TWII', t0_start, t0_end)['Close']

t0_table = pd.DataFrame(index = ticker_list, columns = ['alpha', 'beta', 'skew', 'kurt', 'IRR'])
for i in t0_table.index:
    res = ols(R(data_t0[i]),R(data_market))
    t0_table.loc[i,'alpha'] = res.params.iloc[0]
    t0_table.loc[i, 'beta'] = res.params.iloc[1]
    t0_table.loc[i, 'skew'] = R(data_t0[i]).skew()
    t0_table.loc[i,'kurt'] = R(data_t0[i]).kurt()
    t0_table.loc[i,'IRR'] = IRR(data_t0[i])
t0_table.to_excel('FAI08a_股票數據分析.xlsx')

factors = ['alpha', 'beta', 'skew', 'kurt']
model_t0 = ols(IRR(data_t0),t0_table[factors])
model_t1 = ols(IRR(data_t1),t0_table[factors])

t_vs_t1_table = pd.DataFrame(index = factors, 
                             columns = ['coef_t0','p-value_t0','coef_t1','p-value_t1'])
t_vs_t1_table['coef_t0'] = model_t0.params[factors]
t_vs_t1_table['p-value_t0']  = model_t0.pvalues[factors]
t_vs_t1_table['coef_t1'] = model_t1.params[factors]
t_vs_t1_table['p-value_t1'] = model_t1.pvalues[factors]

t_vs_t1_table.to_excel('FAI08b_股票數據分析.xlsx')