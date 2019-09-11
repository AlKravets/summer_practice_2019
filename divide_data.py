import numpy as np
import math
import random

def divide_data(data, persent):
    train_data = []
    test_data = []
    for row in data:
        if random.random() < persent:
            test_data.append(row)
        else:
            train_data.append(row)
    return test_data, train_data

if __name__ == '__main__':
    import Create_data
    #data = Create_data.create_first_data(40,3)
    data = Create_data.iris_data()
    test_data, train_data = divide_data(data, 0.8)
    print(len(test_data), '  ', test_data[1])
    print(train_data)