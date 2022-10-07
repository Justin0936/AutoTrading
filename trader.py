# You can write code above the if-main block.
#Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

def load_data(args):
  #只要open high
  #training_data= pd.read_csv('.\\' + args.training, header = None)
  #testing_data = pd.read_csv('.\\' + args.testing, header = None)
  training_data= pd.read_csv('.\\' + args.training, names=["Open", "High", "Low", "Close"])
  print (training_data)
  testing_data = pd.read_csv('.\\' + args.testing, names=["Open", "High", "Low", "Close"])
  print (testing_data)
  #train_set = training_data[0]
  #test_set = pd.read_csv(testing_data, header = None)
  #print (args.training)
  #print (training_data)
  #print (train_set)

if __name__ == "__main__":
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    training_data = load_data(args)
    # trader = Trader()
    # trader.train(training_data)

    # testing_data = load_data(args.testing)
    # with open(args.output, "w") as output_file:
    #     for row in testing_data:
    #         # We will perform your action as the open price in the next day.
    #         action = trader.predict_action(row)
    #         output_file.write(action)

    #         # this is your option, you can leave it empty.
    #         trader.re_training()
