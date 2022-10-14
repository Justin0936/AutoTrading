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
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

cust_callback = [
    EarlyStopping(monitor='val_loss', patience=30, verbose=2)
    ]

      

def LSTM_Mode(X_train, y_train, input_length, input_dim):
    # Setting Model
    d = 0.3
    model = Sequential()
    model.add(LSTM(256, input_shape=(input_length, input_dim),
                   return_sequences=True, dropout=0.3, recurrent_dropout=d))
    model.add(Dropout(d))
    # model.add(LSTM(128, input_shape=(input_length, input_dim),
    #                return_sequences=True, dropout=0.25, recurrent_dropout=d))
    # model.add(Dropout(d))
    # model.add(LSTM(64, input_shape=(input_length, input_dim),
    #                return_sequences=True, dropout=0.2, recurrent_dropout=d))
    # model.add(Dropout(d))
    # model.add(LSTM(16, input_shape=(input_length, input_dim),
    #                 return_sequences=False, dropout=0.1, recurrent_dropout=d))
    # model.add(Dropout(d))

    # linear / softmax(多分類) / sigmoid(二分法)
    # model.add(Dense(1, activation='linear'))
    model.add(keras.layers.TimeDistributed(Dense(1, activation='linear')))
    
    # optimizer = tf.keras.optimizers.Adam(lr=0.00005)
    # model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    # loss=mse/categorical_crossentropy
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # fit
    # model = Sequential()
    # model.add(LSTM(input_shape=(None, 1), units=8, unroll=False))
    # model.add(Dense(units=2))
    # # model.compile(optimizer='adam', loss='mean_squared_error',
    # #               metrics=['accuracy'])
    # opt = keras.optimizers.Adam(learning_rate=0.01)
    # model.compile(optimizer=opt, loss='mse',
    #               metrics=['accuracy'])
    # # https://keras.io/zh/models/sequential/
    # # model.fit(X_train, y_train, batch_size=10, nb_epoch=200)
    # # model.fit(X_train, y_train, batch_size=8, epochs=100,
    # #           validation_split=0.2, verbose=2,
    # #           callbacks=cust_callback)
    
    model.summary()
    history = model.fit(X_train, y_train, batch_size=10, epochs=32,
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


def preProcessData(training_data, test_data, sc_x, sc_y, sequence_length):
    temp_set = pd.concat([training_data, test_data], axis = 0)
    # print (temp_set)
    Data_Length =len(temp_set)
    #print("Data_Length:" + str(Data_Length))
    
    # training_data = temp_set[0:-20]
    # print (training_data)
    # testing_data = temp_set[-20:]
    # print (testing_data)
    '''
    Use MinMaxScaler
    Transform features by scaling each feature to a given range.
     This estimator scales and translates each feature individually
      such that it is in the given range on the training set,
      e.g. between zero and one.
    '''
    # Rescale Data
    # scaler.fit(pre_ConcatData)
    # sc_x.partial_fit(training_data)
    # sc_x.partial_fit(test_data)
    # sc_y.partial_fit([test_data["Close"]])
    # sc_y.partial_fit([training_data["Close"]])
    # sc_y.fit([np.concatenate([test_data["Close"], training_data["Close"]])])
    
    # sc_x.fit(temp_set)
    # sc_y.fit([temp_set["Close"]])
    sc_x.fit(training_data)
    sc_y.fit([training_data["Close"]])
    Data_Length =len(training_data)
    
    # sc_y.fit(temp_set["Close"].values.reshape(-1,1))
    # print(temp_set.iloc[:,3].values.reshape(-1,1))    
    # print(training_data["Close"].values.reshape(-1,1))
    
    X_normalize = sc_x.fit_transform(training_data)
    df_X_normalize = pd.DataFrame(X_normalize,
                                  columns=["Open", "High", "Low", "Close"])

    # scaler.partial_fit(training_data["Open"])
    # scaler.partial_fit(training_data["High"])
    # scaler.partial_fit(training_data["Low"])
    # scaler.partial_fit(training_data["Close"])
    # scaler.partial_fit(test_data["Open"])
    # scaler.partial_fit(test_data["High"])
    # scaler.partial_fit(test_data["Low"])
    # scaler.partial_fit(test_data["Close"])

    # X_normalize = scaler.fit_transform(training_data["Close"])
    # df_X_normalize = pd.DataFrame(X_normalize,
    #                               columns=["Close"])
    # print(df_X_normalize)

    # X_normalize = scaler.fit_transform(test_data)
    # df_X_normalize = pd.DataFrame(X_normalize,
    #                               columns=["Open", "High", "Low", "Close"])
    # print(df_X_normalize)

    # Select feature Set train set
    train_set = df_X_normalize.iloc[:, 3]
    # print(train_set)

    # preparation sequence Data
    # sequence_length = 10
    data = []
    for i in range(len(train_set) - sequence_length):
        data.append(train_set[i: i + sequence_length + 1])
    # print(data)

    reshaped_data = np.array(data)
    x = reshaped_data[:, :-1]
    # print(x)
    y = reshaped_data[:, -1]
    # print(y)
    ''' Spilt Data (train set and test set) '''
    split_boundary = int(reshaped_data.shape[0] * 0.8)
    # print(split_boundary)
    train_x = x[: split_boundary]
    # print(train_x)
    test_x = x[split_boundary:]
    # print(test_x)
    train_y = y[: split_boundary]
    # print(test_x)
    test_y = y[split_boundary:]
    # print(test_y)
    return train_x, test_x, train_y, test_y, Data_Length


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
    DayTime_Step = 10
    # Read CSV Dataset
    training_data = pd.read_csv('.\\' + args.training,
                                names=["Open", "High", "Low", "Close"])
    test_data = pd.read_csv('.\\' + args.testing,
                            names=["Open", "High", "Low", "Close"])
    # data_visualization(training_data, test_data)
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    train_x, test_x, train_y, test_y, Data_Length = preProcessData(
        training_data, test_data, scaler_x, scaler_y, DayTime_Step)
    
    # # setting & compile & fit model
    model_1, history = LSTM_Mode(train_x, train_y, DayTime_Step, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss by LSTM')
    plt.legend()
    plt.show()
    
    # For Test Use    
    model_1 = keras.models.load_model('./LTMS_mode.h5')
    
    # print(test_x)
    # # Process testing Data
    # predict = model_1.predict(test_x)
    # print(predict)
    # print(len(predict))
    # predict = scaler_y.inverse_transform(predict)
    # test_y = scaler.inverse_transform(test_y)
    
    # plt.figure(figsize=(12,6))
    # plt.plot(predict, 'b-')
    # plt.plot(test_y, 'r-')
    # plt.legend(['predict', 'realdata'])
    # plt.show()
    
    # data_Analysis(training_data)

    # read Test data
    # testing_data = load_data(args.testing)
    predict_data = np.zeros(shape=(1, Data_Length))
    # print(len(predict_data[0]))
    Data = training_data['Close'][-DayTime_Step:].reset_index (drop = True)
    # print (Data)

    # predict_data[:0] = [training_data['Close'][-DayTime_Step:].to_numpy()]
    for i in range(len(Data)):
        # print (Data[i])
        predict_data[0][i] = Data[i]
        # print(predict_data[0][i])
        
    # -------------    
    # predict_data = scaler_y.transform(predict_data)
    # predict_data = [predict_data[-DayTime_Step:].to_numpy()]
    predict_data = np.delete(predict_data, np.s_[DayTime_Step:], axis=1)
    print(predict_data)
    print(type(predict_data))
    predict = model_1.predict(predict_data)
    print(predict)
    
    train_predict_new = np.zeros(shape=(len(predict), Data_Length))
    train_predict_new[:, 0] = predict[:, 0]
    print(train_predict_new)
    trainPredict = scaler_y.inverse_transform(train_predict_new)[:, 0]
    print(trainPredict)
    
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