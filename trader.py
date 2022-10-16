if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    import pandas as pd
    import training_module

    data = pd.read_csv('testing_data.csv', header=None,
                       names=['Open', 'High', 'Low', 'Close'])
    f = open('output.csv', 'w')
    # 第一天不動作0,最後兩天不讀取數值(最後一天歸0)
    action = 0
    f.write('0\n')
    data.drop(data.tail(2).index, inplace=True)
    P = 0.9

    # 策略判斷
    for i, j, m, n in zip(data['High'], data['Low'], data['Open'],
                          data['Close']):
        if (i - j) / m * 100 > P * training_module.mean():
            if m > n:  # 買進
                if action == 0:
                    action += 1
                    f.write('1\n')
                elif action == -1:
                    action += 1
                    f.write('1\n')
                elif action == 1:
                    action = action
                    f.write('0\n')
            elif m < n:  # 賣出
                if action == 0:
                    action -= 1
                    f.write('-1\n')
                elif action == -1:
                    action = action
                    f.write('0\n')
                elif action == 1:
                    action -= 1
                    f.write('-1\n')
            elif m == n:  # 不動
                action = action
                f.write('0\n')
        else:
            action = action
            f.write('0\n')

    f.close()
