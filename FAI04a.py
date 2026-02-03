'''1. 9 的 6 次方如何撰寫程式? 有一個串列內含[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]，請利用 if 指令取
出其中的偶數。
2. 如果某人有買台積電股票，則此投資人可稱為'科技人' ，如果某人有買中華電股票，則
此投資人可稱為'電信人' ，如果兩種股票都有購買，則此投資人也可稱為'科技人與電信
人' 。請撰寫程式判斷，甲購買了：台積電、台泥、中鋼、中華電、台塑化，則甲可稱
為何種人? (註: if … in)
3. 某投資人期初買入上櫃市值最高的 10 檔股票，之後賣出[‘5274’, ‘3529’]，請輸出此投資
人持有的剩餘股票代碼到 FAI04a.xlsx。(注意代碼)
4. 請問台灣上市、上櫃股票各市值最高前 10 檔(共 20 檔)在 20250923~20251007 期間，以股
票之調整後收盤價計算累績報酬率(IRR)，將結果輸出到到 FAI04b.xlsx，請問: (1)有哪
幾檔股票的 IRR 超越台灣 50?，請輸出到第 1 個工作表:【"Sheet1"】 (2)哪一檔股票在
這段期間的累積報酬率(IRR)最高，IRR 為何? ，請輸出到第 2 個工作表:【"Sheet2"】。
(IRR=[ST-S0]/S0)
1 ~ 3 每題 15%，1 ~2 題由 Spyder 輸出即可。
第 4 題 55%
'''

#1
print(pow(9, 6))

a = list(range(1,12))
even_numbers = []

for i in a:
    if i % 2 ==0:
        even_numbers.append(i)

print(even_numbers)

#2
stocks = ['台積電', '台泥', '中鋼', '中華電', '台塑化']
if '中華電' in stocks and '台積電' in stocks:
    print('甲是電信人和科技人')
elif '台積電' in stocks:
    print('甲是科技人')
elif '中華電' in stocks:
    print('甲是電信人')
else:
    print('甲不是科技人也不是電信人')
    
#3
import pandas as pd
TW_PickStockNo = 10
TW_stock_ticker =  pd.read_html('https://www.taifex.com.tw/cht/2/tPEXPropertion' , encoding='utf-8')[0]['證券名稱'][:TW_PickStockNo]
TW_stock_ticker_list = [str(i) for i in TW_stock_ticker]
sold = ['5274', '3529']
TW_stock_ticker_list = [i for i in TW_stock_ticker_list if i not in sold]
pd.DataFrame({'剩餘持股': TW_stock_ticker_list}).to_excel('FAI04a.xlsx', index=True)
