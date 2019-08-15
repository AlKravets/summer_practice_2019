import numpy as np
import math
import os
import pylab as pl
from matplotlib.colors import ListedColormap
import random


def create_first_data(size_class,number_of_classes):
    data = []
    for classNum in range(number_of_classes):
        #Choose random center of 2-dimensional gaussian
        centerX, centerY = random.random()*5.0, random.random()*5.0
        #Choose numberOfClassEl random nodes with RMS=0.5
        for rowNum in range(size_class):
            data.append([ [random.gauss(centerX,0.5), random.gauss(centerY,0.5)], classNum])
    return data


def showData (nClasses, nItemsInClass):
    trainData      = create_first_data (nItemsInClass, nClasses)
    classColormap  = ListedColormap(['#FF0000', '#00FF00', '#000000'])
    pl.scatter([trainData[i][0][0] for i in range(len(trainData))],
               [trainData[i][0][1] for i in range(len(trainData))],
               c=[trainData[i][1] for i in range(len(trainData))],
               cmap=classColormap)
    pl.show()   


if __name__ == '__main__':
    showData (3, 40)