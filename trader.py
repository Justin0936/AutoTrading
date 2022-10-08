# You can write code above the if-main block.
# Import
import pandas as pd
# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# def load_data(args):
#   #只要open high
#   training_data= pd.read_csv('.\\' + args.training, header = None)
#   testing_data = pd.read_csv('.\\' + args.testing, header = None)
#   training_data= pd.read_csv('.\\' + args.training,
#     names=["Open", "High", "Low", "Close"])
#   print (training_data)
#   testing_data = pd.read_csv('.\\' + args.testing,
#     names=["Open", "High", "Low", "Close"])
#   print (testing_data)
#   #train_set = training_data[0]
#   #test_set = pd.read_csv(testing_data, header = None)
#   #print (args.training)
#   #print (training_data)
#   #print (train_set)

# def data_Analysis(df):
    # # 可以用 matplotlib 做出多張子圖，再用 seaborn 畫在這些子圖上
    # fig, axis = plt.subplots(1, 3, figsize=(16, 12))

    # # ax 傳進上面做出的子圖，axis[1,2] 就代表要畫在第 1 個 row，第 2 個 column 這張子圖上
    # sns.boxplot(x = "Open", y = "High", data = df, ax = axis[0])
    # sns.boxplot(x = 'Open', y = 'Low', data=df, ax = axis[1])
    # sns.boxplot(x = 'Open', y = 'Close', data=df, ax = axis[2])
    
    # plt.tight_layout()
    # plt.show()
    
    # print(df.corr())
    # #相關係數, 用熱點圖來觀察
    # plt.figure(figsize=(10, 8))
    # data_corr = df.corr()
    # #熱點圖最大值為1, 最小值為0.7, 將數值顯示在熱點圖上
    # sns.heatmap(data_corr, vmax = 1, vmin = 0.9999999, square=True, annot=True)
    # plt.show()
    # fig, axis = plt.subplots(1, 3, figsize=(16, 12))
    # ax = df.Open.plot(kind='kde')
    # df.Open.plot(kind='hist', bins=40, ax=ax)
    # df.Open.plot(kind='kde', ax=ax, secondary_y=True)
    # ax.set_xlabel('Open')
    
    # ax_High = df.High.plot(kind='kde')
    # df.High.plot(kind='hist', bins=40, ax=ax_High)
    # df.High.plot(kind='kde', ax=ax_High, secondary_y=True)
    # ax_High.set_xlabel('High')
    
if __name__ == "__main__":
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv",
                        help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv",
                        help="input testing data file name")
    parser.add_argument("--output", default="output.csv",
                        help="output file name")
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    # Read CSV Dataset
    training_data = pd.read_csv('.\\' + args.training,
                                names=["Open", "High", "Low", "Close"])
    data_Length = training_data.shape[0]
    # print(type(source_X))
    # data_Analysis(training_data)
    df = training_data.iloc[:, 0:4].values
    # print (df)
    df = training_data[["Open", "High", "Low", "Close"]]
    # print (df)
    X_preproc = df[["High", "Low", "Close"]]
    y_preproc = df["Open"]
    '''
    Use MinMaxScaler
    Transform features by scaling each feature to a given range.
     This estimator scales and translates each feature individually
      such that it is in the given range on the training set,
      e.g. between zero and one.
    '''
    scaler = MinMaxScaler()
    scaler.fit(X_preproc)
    # print (df)
    X_normalize = scaler.fit_transform(X_preproc)
    df_X_normalize = pd.DataFrame(X_normalize,
                                  columns=["High", "Low", "Close"])
    # print(df_X_normalize)
    # X_train, X_valid, y_train, y_valid = train_test_split(
    #     df_X_normalize, y_preproc, random_state=9527)
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        df_X_normalize, y_preproc, train_size=0.8,
        random_state=None, shuffle=False)
    
    # Start Traing Mode
    print(X_train)
    # data_Analysis(training_data)


    # print (df)
    # print (df.size)
    
    # scaler.fit( training_data["Open"] )    
    # training_data = load_data(args)
    # trader = Trader()
    # trader.train(training_data)
    # read
    # testing_data = load_data(args.testing)
    # with open(args.output, "w") as output_file:
    #     for row in testing_data:
    #         # We will perform your action as the open price in the next day.
    #         action = trader.predict_action(row)
    #         output_file.write(action)

    #         # this is your option, you can leave it empty.
    #         trader.re_training()
