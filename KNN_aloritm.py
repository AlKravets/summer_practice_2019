import numpy as np
import math
import os
import time
import random
import Create_data
import divide_data
import distance_between_elements

random.seed(20)

def KNN(test_data: list, train_data: list, k: int, number_of_clases: int):
    test_lables = []
    for test_point in test_data:
        #print(test_point)
        test_dist = [[distance_between_elements.euvlid_dist(test_point[0], train_data[i][0]), train_data[i][1]] for i in range(len(train_data))]

        stat = [0 for i in range(number_of_clases)]
        for d in sorted(test_dist)[0:k]:
            stat[d[1]] +=1

        test_lables.append((sorted(list(zip(stat,range(number_of_clases))))[-1][1], int(test_point[1])))
    return test_lables

def number_of_errors (test_lables):
    er = 0
    for i in range(len(test_lables)):
        if test_lables[i][1] != test_lables[i][0]:
            er +=1
    return er

if __name__ == '__main__':
    data = Create_data.create_first_data(40,3)
    test_data, train_data = divide_data.divide_data(data, 0.8)
    #Create_data.showData(data)
    lable = KNN(test_data, train_data, 5, 3)
    er = number_of_errors(lable)
    print(er, ' of ', len(lable))