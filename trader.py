# You can write code above the if-main block.
# Import
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline


from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

cust_callback = [
    EarlyStopping(monitor='val_loss', patience=30, verbose=2)
    ]

# --- Global
# Stock_Count=0; {1, 0, -1}
Stock_Count = 0
"""
Stock_Action
forecast = 0; {0:Down ,1:UP}

"""


def Stock_Action(forecast):
    # Stock_Action=0{1:Buy,0:No_Action, -1=Sell}
    global Stock_Count
    Stock_Action = 0

    if forecast == 1:
        # Forecast == 1 (up) Start.
        if Stock_Count >= 0:
            if Stock_Count == 0:
                # is 0
                Stock_Action = 1
                Stock_Count = 1
            else:
                # is 1
                Stock_Action = 0
                Stock_Count = 1
        else:
            # is -1
            Stock_Action = 1
            Stock_Count = 0
        # Forecast == 1 (up) End.
    else:
        # forecase == 0 (down) Start.
        if Stock_Count >= 0:
            if Stock_Count == 0:
                # is 0
                Stock_Action = -1
                Stock_Count = -1
            else:
                # is 1
                Stock_Action = -1
                Stock_Count = 0
        else:
            # is -1
            Stock_Action = 0
            Stock_Count = -1
        # forecase == 0 (down) End.

    return Stock_Action


def LSTM_Mode(X_train, y_train, input_length, input_dim):
    # Clear Session For model
    keras.backend.clear_session()
    d = 0.3
    shape = X_train.shape[1]
    # print (shape)
    # Setting Model
    model = Sequential()
    # model.add(LSTM(50, input_shape=(shape, input_dim),
    #                 return_sequences=True, dropout=0.3, recurrent_dropout=d))
    
    # model.add(LSTM(100, input_shape=(shape, input_dim),
    #                 return_sequences=True))
    # model.add(Dropout(d))
    
    # model.add(LSTM(100, activation='relu'))
    # model.add(Dropout(d))
    
    # ---------------
    # model.add(LSTM(256, input_shape=(shape, input_dim), return_sequences=True))
    # # model.add(Dropout(d))
    # # model.add(LSTM(128, input_shape=(shape, input_dim), return_sequences=True))
    # # model.add(Dropout(d))
    # # model.add(LSTM(64, input_shape=(shape, input_dim), return_sequences=True))
    # # # model.add(Dropout(d))
    # model.add(LSTM(16, input_shape=(shape, input_dim),return_sequences=False))
    # model.add(Dropout(d))
    # ----------------

    # model.add(LSTM(256, input_shape=(shape, input_dim),
    #                 return_sequences=True, dropout=0.3, recurrent_dropout=d))
    # model.add(Dropout(d))
    # model.add(LSTM(128, input_shape=(shape, input_dim),
    #                 return_sequences=True, dropout=0.25, recurrent_dropout=d))
    # model.add(Dropout(d))
    # model.add(LSTM(64, input_shape=(shape, input_dim),
    #                 return_sequences=True, dropout=0.2, recurrent_dropout=d))
    # model.add(Dropout(d))
    # model.add(LSTM(16, input_shape=(shape, input_dim),
    #                 return_sequences=True, dropout=0.2, recurrent_dropout=d))
    # model.add(Dropout(d))
    model.add(LSTM(units = 256, return_sequences = True, input_shape = (shape, input_dim)))
    model.add(LSTM(units = 100, return_sequences = True))
    model.add(Dropout(d))
    model.add(Flatten())
    model.add(Dense(units=128))
    model.add(Dense(units=32))
    # linear / softmax(多分類) / sigmoid(二分法)
    model.add(Dense(1, activation='linear'))
    # model.add(keras.layers.TimeDistributed(Dense(1, activation='linear')))

    # optimizer = tf.keras.optimizers.Adam(lr=0.00005)
    # model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    # loss=mse/categorical_crossentropy
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    
    model.summary()
    history = model.fit(X_train, y_train, batch_size=32, epochs=300,
                        validation_split=0.2, verbose=2, shuffle=False,
                        callbacks=cust_callback)
    model.save('LTMS_mode.h5')
    return model, history


def data_visualization(train, test):
    plt.plot(train.iloc[80:160, 0:1], color='blue', label='open')
    plt.plot(train.iloc[80:160, 1:2], color='red', label='high')
    plt.plot(train.iloc[80:160, 2:3], color='green', label='low')
    plt.plot(train.iloc[80:160, 3:4], color='black', label='close')
    plt.legend()
    plt.xticks(range(80, 170, 10))
    plt.show()

    plt.plot(test.iloc[:, 0], color='blue', label='open')
    plt.plot(test.iloc[:, 1], color='red', label='high')
    plt.plot(test.iloc[:, 2], color='green', label='low')
    plt.plot(test.iloc[:, 3], color='black', label='close')
    plt.legend()
    plt.xticks(range(0, 25, 5))
    plt.show()


def preProcessData(training_data, test_data, sc_x, sequence_length):
    train_set = training_data['Open']
    test_set = test_data['Open']
    # print(train_set)
    train_set = train_set.values.reshape(-1, 1)
    # print(train_set)

    X_normalize = sc_x.fit_transform(train_set)

    train_X = []
    train_y = []
    for i in range(sequence_length, len(train_set)):
        train_X.append(X_normalize[i-sequence_length:i-1, 0])
        train_y.append(X_normalize[i, 0])

    train_X, train_y = np.array(train_X), np.array(train_y)
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    # print(train_X[0])
    # print(y_train[0])

    # Prepare Test predict dataset
    total_set = pd.concat((training_data['Open'], test_data['Open']), axis=0)
    #print(total_set)
    test_X = total_set[len(total_set) - len(test_set)
                       - sequence_length:].values
    # print(test_X)
    test_X = test_X.reshape(-1, 1)
    # print(test_X)
    test_X = sc_x.transform(test_X)
    
    return train_X, train_y, test_X


def Test_Ouptdat():
    # with open(args.output, 'w') as output_file:
    #     for i in range(len(action)):
            X_test = []
            for i in range(DayTime_Step, len(test_X)):
                X_test.append(test_X[i-DayTime_Step:i-1, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                # print(X_test)
                # predicted_stock_price = model_1.predict(X_test)
                # #使用sc的 inverse_transform將股價轉為歸一化前
                # predicted_stock_price = scaler_x.inverse_transform(predicted_stock_price)
                # output_file.writelines(str(action[i])+"\n")


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
    ''' We use path 3 days price to predict the next day '''
    DayTime_Step = 5
    # Read CSV Dataset
    training_data = pd.read_csv('.\\' + args.training,
                                names=["Open", "High", "Low", "Close"])
    test_data = pd.read_csv('.\\' + args.testing,
                            names=["Open", "High", "Low", "Close"])
    # data_visualization(training_data, test_data)
    scaler_x = MinMaxScaler(feature_range=(0, 1))

    train_x, train_y, test_X = preProcessData(training_data, test_data,
                                              scaler_x, DayTime_Step)

    # setting & compile & fit model
    model_1, history = LSTM_Mode(train_x, train_y, DayTime_Step, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Loss by LSTM')
    plt.legend()
    plt.show()

    # For Test Use
    model_1 = keras.models.load_model('./LTMS_mode.h5')

    X_test = []
    for i in range(DayTime_Step, len(test_X)):
        X_test.append(test_X[i-DayTime_Step:i-1, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # print(X_test)
    predicted_stock_price = model_1.predict(X_test)
    # 使用sc的 inverse_transform將股價轉為歸一化前
    predicted_stock_price = scaler_x.inverse_transform(predicted_stock_price)

    plt.plot(test_data['Open'].values, color='black',
             label='Real Test Stock Price')
    plt.plot(predicted_stock_price, color='green',
             label='Predicted Stock Price')
    plt.title('Stock Prediction')
    plt.xlabel('Time(days)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

        
    # # -------------    
    # # predict_data = scaler_y.transform(predict_data)
    # # predict_data = [predict_data[-DayTime_Step:].to_numpy()]
    # predict_data = np.delete(predict_data, np.s_[DayTime_Step:], axis=1)
    # print(predict_data)
    # print(type(predict_data))
    # predict = model_1.predict(predict_data)
    # print(predict)
    
    # train_predict_new = np.zeros(shape=(len(predict), Data_Length))
    # train_predict_new[:, 0] = predict[:, 0]
    # print(train_predict_new)
    # trainPredict = scaler_y.inverse_transform(train_predict_new)[:, 0]
    # print(trainPredict)
    
    # print(Data)
    # predict = model_1.predict(Data)
    # print(predict)
    # print(scaler_y.inverse_transform(predict)[:, 0])
    
    # --------------------------------------
    # # predict_data =[-DayTime_Step:]
    # test_data_close = test_data['Close']
    # # print(test_data_close)
    # # print(type(test_data_close))
    # with open(args.output, "w") as output_file:
    #     # for (col_Name, col_Data) in test_data.iteritems():
    #     # for column in test_data[["Close"]]:
    #     # for index in test_data.index:
    #     for i in range(len(test_data_close) -1 ):
    #         # We will perform your action as the open price in the next day.
    #         # print(test_data_close[i])
            
    #         train_predict_new = np.zeros(shape=(1, Data_Length))
    #         train_predict_new[:, 0] = test_data_close[i]
    #         # print(train_predict_new)
    #         trainPredict = scaler_y.transform(train_predict_new)[:, 0]
    #         # print(trainPredict)
            
    #         predict = model_1.predict(np.array(trainPredict, ndmin=2))
    #         # predict = model_1.predict(np.array(test_data_close[i], ndmin=2))
    #         train_predict_new = np.zeros(shape=(len(predict), Data_Length))
    #         train_predict_new[:, 0] = predict[:, 0]
    #         trainPredict = scaler_y.inverse_transform(train_predict_new)[:, 0]
    #         print(trainPredict)


    #         # action = trader.predict_action(row)
    # #         output_file.write(action)

    # #         # this is your option, you can leave it empty.
    # #         trader.re_training()