import numpy as np
import math
import os
import pylab as pl
from matplotlib.colors import ListedColormap
import random
import urllib
import csv


def create_first_data(size_class,number_of_classes):
    data = []
    for classNum in range(number_of_classes):
        #Choose random center of 2-dimensional gaussian
        centerX, centerY = random.random()*5.0, random.random()*5.0
        #Choose numberOfClassEl random nodes with RMS=0.5
        for rowNum in range(size_class):
            data.append([ [random.gauss(centerX,0.5), random.gauss(centerY,0.5)], classNum])
    return data


def showData (trainData):
    #trainData      = create_first_data (nItemsInClass, nClasses)
    classColormap  = ListedColormap(['#FF0000', '#00FF00', '#000000'])
    pl.scatter([trainData[i][0][0] for i in range(len(trainData))],
               [trainData[i][0][1] for i in range(len(trainData))],
               c=[trainData[i][1] for i in range(len(trainData))],
               cmap=classColormap)
    pl.show()


def iris_data():
    l = []

    with open('iris.data','r') as f:
        for line in f:
            l1 = []
            for word1 in line.split():
                for word in word1.split(','):
                    l1.append(word)
            l.append(l1)
    

    #print(len(l))
    data = []
    for i in range(len(l)):
        l[i][0] = float(l[i][0])
        l[i][1] = float(l[i][1])
        l[i][2] = float(l[i][2])
        l[i][3] = float(l[i][3])
        if l[i][4] == "Iris-setosa":
            l[i][4] = 0
        elif l[i][4] == "Iris-versicolor":
            l[i][4] = 1
        elif l[i][4] == "Iris-virginica":
            l[i][4] = 2
        else:
            print("error")
        data.append([[l[i][0],l[i][1],l[i][2],l[i][3]], l[i][4]])
    return data

if __name__ == '__main__':
    #data = create_first_data(40,3)
    #showData (data)
    
    data = iris_data()
    
    
