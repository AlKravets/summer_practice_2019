import numpy as np
import math
import random


def euvlid_dist(a: list,b: list):
    res = 0
    for i in range(len(a)):
        res += (a[i] - b[i])**2
    
    return math.sqrt(res)

if __name__ == '__main__':
    import Create_data
    import divide_data
    data = Create_data.create_first_data(40,3)
    test_data, train_data = divide_data.divide_data(data, 0.8)
    print(euvlid_dist(test_data[0][0],test_data[1][0]))

    print(test_data[0][0],'   ', test_data[1][0])