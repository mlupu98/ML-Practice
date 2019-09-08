import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import tensorflow
import keras
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#uses panda to read csv file
def read_data(filename, columns):

    #read file into dataframe
    data = pd.read_csv(filename, sep=",")

    #selectedData = data[["Result", "FSP.1", "ACE.1", "FSW.1", "BPC.1", "BPW.1", "NPA.1", "NPW.1", "TPW.1"]]
    #selectedData = data[["Result", "BPW.1", "FSW.1", "TPW.1"]]

    #choose columns of interest
    selectedData = data[columns]

    #number of data entries
    size = len(data[columns[0]])

    #drop all NaN
    correctedData = selectedData.dropna()

    return correctedData

#predict is the variable that you want to predict
def train_model(correctedData, predict):

    #seperate prediction variable from the rest
    X = np.array(correctedData.drop([predict], 1))
    Y = np.array(correctedData[predict])

    highestAccuracy = 0

    #find the most accurate training model
    for i in range(20):

        #train the model on 90% of the data
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

        #run linear regression
        linear = linear_model.LinearRegression()

        linear.fit(x_train, y_train)

        accuracy = linear.score(x_test, y_test)

        #check accuracy and replace results if this run has better results
        if accuracy > highestAccuracy:

            print("Accuracy: \n", accuracy)

            with open("tennisresults.pickle", "wb") as f:
                pickle.dump(linear, f)

    return x_test, y_test

#print regression results
def print_results(x_test, y_test):


    pickle_in = open("tennisresults.pickle", "rb")
    linear = pickle.load(pickle_in)


    print("Coefficient: \n", linear.coef_)
    print("Intercept: \n", linear.intercept_)

    predictions = linear.predict(x_test)

    for x in range(len(predictions)):
        print(predictions[x], x_test[x], y_test[x])

    return

#graph results
def graph_results(correctedData, xplot, yplot):

    style.use("ggplot")
    plt.scatter(correctedData[xplot], correctedData[yplot])

    plt.xlabel(xplot)
    plt.ylabel(yplot)
    plt.show()

    return


def main():

    columns = ["Result", "BPC.1", "BPC.2"]

    data = read_data("Tennis-Major-Tournaments-Match-Statistics/AusOpen-women-2013.csv", columns)

    x_test, y_test = train_model(data)

    print_results(x_test, y_test)

    xplot = "BPC.1"
    yplot = "Result"

    graph_results(data, xplot, yplot)

    return

if __name__ == '__main__':
    main()