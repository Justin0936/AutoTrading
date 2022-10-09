# You can write code above the if-main block.
# Import
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping


cust_callback = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=2)
    ]


def LSTM_Mode(X_train, y_train):
    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    
    model = Sequential()
    model.add(LSTM(input_shape=(None, 1), units=8, unroll=False))
    model.add(Dense(units=2))
    # model.compile(optimizer='adam', loss='mean_squared_error',
    #               metrics=['accuracy'])
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='mse',
                  metrics=['accuracy'])
    # https://keras.io/zh/models/sequential/
    # model.fit(X_train, y_train, batch_size=10, nb_epoch=200)
    model.fit(X_train, y_train, batch_size=8, epochs=100,
              validation_split=0.2, verbose=2,
              callbacks=cust_callback)
    model.save('LTMS_mode.h5')
    return model


def Verify_and_TestPredict(predict_data, y_valid):
    # predict_data = model.predict(X_train)
    # predict = scaler.inverse_transform(predict)
    # verif_y = scaler.inverse_transform(y_valid)
    plt.figure(figsize=(12, 6))
    plt.plot(predict_data, 'b-')
    plt.plot(y_valid, 'r-')
    plt.legend(['predict', 'realdata'])
    plt.show()


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
    test_data = pd.read_csv('.\\' + args.testing,
                            names=["Open", "High", "Low", "Close"])
    data_Length = training_data.shape[0]
    # print(type(source_X))
    # data_Analysis(training_data)
    df = training_data.iloc[:, 0:4].values
    # print (df)
    df = training_data[["Open", "High", "Low", "Close"]]
    df_Open = training_data[["Open"]]
    df_High = training_data[["High"]]
    df_Low = training_data[["Low"]]
    df_Close = training_data[["Close"]]
    # print (df)
    X_preproc = df[["High", "Low", "Close"]]
    y_preproc = df["Open"]

    test_preproc = test_data["Open"]
    '''
    Use MinMaxScaler
    Transform features by scaling each feature to a given range.
     This estimator scales and translates each feature individually
      such that it is in the given range on the training set,
      e.g. between zero and one.
    '''
    # Rescale Data
    scaler = MinMaxScaler()
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler.fit(X_preproc)
    # scaler.partial_fit(df_Open)
    scaler.partial_fit(df_High, df_Open)
    scaler.partial_fit(df_Low, df_Open)
    scaler.partial_fit(df_Close, df_Open)
    # print (df)
    # X_normalize = scaler.fit_transform(X_preproc)
    # df_X_normalize = pd.DataFrame(X_normalize,
    #                               columns=["High", "Low", "Close"])
    df_High_normalize = scaler.fit_transform(df_High)
    df_Low_normalize = scaler.fit_transform(df_Low)
    df_Close_normalize = scaler.fit_transform(df_Close)
    # print(df_High_normalize)

    X_normalize = np.concatenate([df_High_normalize, df_Low_normalize,
                                 df_Close_normalize], axis=1)

    # X_normalize = pd.concat([df_High_normalize, df_Low_normalize,
    #                           df_Close_normalize], axis=1)
    # print(X_normalize)

    # try pre columns=["High", "Low", "Close"]
    df_X_normalize = pd.DataFrame(X_normalize,
                                  columns=["High", "Low", "Close"])
    # print(df_X_normalize)
    # Rescale Data
    
    
    # X_train, X_valid, y_train, y_valid = train_test_split(
    #     df_X_normalize, y_preproc, random_state=9527)

    X_train, X_valid, y_train, y_valid = train_test_split(
        df_X_normalize, y_preproc, train_size=0.8,
        random_state=None, shuffle=False)
    #print(X_train)
    
    
    # Start Traing Mode
    # method 1
    Mode_1 = LSTM_Mode(X_train, y_train)
    Mode_1 = load_model('LTMS_mode.h5')
    # predict_data = X_train[["High", "Low", "Close"]]
    # predict_data = Mode_1.predict(X_train)
    predict_data = Mode_1.predict(X_train)
    # print(predict_data)

    predict_data = scaler.inverse_transform(predict_data)
    print(predict_data)
    # print(y_train)


    # data_Analysis(training_data)

    
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
