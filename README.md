# NCKU-MLDS homework1

## trader.py 判斷策略

>當股價出現**大幅度漲跌**時才進行買賣動作

>第一天不進行買賣動作(action= 0),
第二天開始判斷前一天的最高點與最低點差距百分比,
當差距大於訓練資料集差距百分比之平均時,才進行買賣

>前一天開盤價大於收盤價==>判斷要買進 , 開盤價小於收盤價==>判斷要賣出 , 開盤價等於收盤價==>不進行買賣動作


## training_module.py
>為訓練資料集的高低點差距百分比之平均值

## output.csv
為每天輸出的買賣動作


## Partner Synchronize Collaborate:
https://github.com/Ophelialuh/NCKU-MLDS
