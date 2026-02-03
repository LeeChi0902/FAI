'''
請上網抓取 20250102~20250630，下載台灣大盤前 150 檔股票、的調整後收盤價，計算
股價報酬率估計每一檔股票的 {α、β、std、Skew、Kurt、IRR}，每一檔股票會有五組
參數，如下列表格所示，輸出到 FAI09.xlsx
以 {α、β}、 {α、Skew}、 {β、Skew} 當成 Kmeans (k=3) 的分類特徵值，計算各種分類
之後的各三組資料之中各檔股票的: IRR 的均值，查驗是否有明顯的差異。
例如: {α、β} 輸出到: FAI09_AB.xlsx
{α、Skew}輸出到: FAI09_AS.xlsx
{β、Skew}輸出到: FAI09_BS.xlsx
'''
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.cluster import KMeans

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

start = '2025-01-02'
end = '2025-07-01'
stock_ticker = pd.read_html('https://www.taifex.com.tw/cht/9/futuresQADetail' , encoding='utf-8')[0]['證券名稱'][:150]
ticker_list = [str(i)+'.TW' for i in stock_ticker]
stock_data = yf.download(ticker_list,start,end)['Close'].dropna(axis=1,how='all')
market_data = yf.download('^TWII',start,end)['Close']

factors = ['α','β','std','Skew','Kurt','IRR']
table = pd.DataFrame(index=stock_data.columns, columns=factors)
for i in table.index:
    r = R(stock_data[i])
    res = ols(r,R(market_data))
    table.loc[i,'α'] = res.params.iloc[0]
    table.loc[i,'β'] = res.params.iloc[1]
    table.loc[i,'std'] = r.std(ddof = 1)
    table.loc[i,'Skew'] = r.skew()
    table.loc[i,'Kurt'] = r.kurt()
    table.loc[i,'IRR'] = IRR(stock_data[i])
table.to_excel('FAI09_股票數據分析.xlsx')

for i in [['α','β'],['α','Skew'],['β','Skew']]:
    X = table[i]
    Y = table['IRR']
    kms = KMeans(n_clusters=3, random_state=42).fit(X)
    group = pd.DataFrame(kms.labels_, columns=['group'], index=X.index)
    Kmeans_m = pd.concat([Y, group], axis=1, join='inner')
    K_group_IRR = Kmeans_m.groupby(['group']).mean()
    K_group_IRR.to_excel(f'FAI09{i}group.xlsx')