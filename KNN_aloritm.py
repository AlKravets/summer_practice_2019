import numpy as np
import math
import os
import time
import random

random.seed(20)

name_file_BC = 'results_ver2_BC.txt'
name_file_Control = 'results_ver2_Control.txt'
name_file_Fam = 'results_ver2_Fam.txt'




data_BC = np.loadtxt(name_file_BC, delimiter='\n', dtype= np.float)
data_Control = np.loadtxt(name_file_Control, delimiter='\n', dtype= np.float)
data_Fam = np.loadtxt(name_file_Fam, delimiter='\n', dtype= np.float)

print('BC shape:', data_BC.shape[0], 'control shape:', data_Control.shape[0] , 'Fam shape:', data_Fam.shape[0], sep = ' ', end = '\n')




centerX= random.random()*5.0
centerX= 1
for i in range(3326):

    data_BC[i] = random.gauss(centerX,0.5)    
centerX= random.random()*5.0
centerX= 3
for i in range(1597):
    data_Control[i] = random.gauss(centerX,0.5)  
centerX= random.random()*5.0
centerX= random.random()*5.0
centerX= 5
for i in range(1670):
    data_Fam[i] = random.gauss(centerX,0.5)  


numbers = 100
k = 7

train_data_number= np.zeros((3, numbers), dtype = np.int64)

train_data_number[0] = random.sample(range(0,data_BC.shape[0]), numbers)
train_data_number[1] = random.sample(range(0,data_Control.shape[0]), numbers)
train_data_number[2] = random.sample(range(0,data_Fam.shape[0]), numbers)


train_data= np.zeros((3, numbers))

for i in range(numbers):
    train_data[0][i] = data_BC[train_data_number[0][i]]
    train_data[1][i] = data_Control[train_data_number[1][i]]
    train_data[2][i] = data_Fam[train_data_number[2][i]]

New_BC = 0
BC_error = 0
New_Control = 0
Control_error = 0
New_Fam = 0
Fam_error =0
All_error =0

for i in range(data_BC.shape[0]):
    dist  = np.zeros((3*numbers, 2))
    for j in range(numbers):
        dist[3*j][0] = abs(data_BC[i] - train_data[0][j])
        dist[3*j][1] = 0
        dist[3*j+1][0] = abs(data_BC[i] - train_data[1][j])
        dist[3*j+1][1] = 1
        dist[3*j+2][0] = abs(data_BC[i] - train_data[2][j])
        dist[3*j+2][1] = 2
    
    dist = dist[dist[:,0].argsort(kind='mergesort')]
    
    bc_l =0
    Cont_l =0
    Fam_l =0
    for j in range(k):
        if dist[j][1] == 0:
            bc_l+=1
        if dist[j][1] == 1:
            Cont_l+=1
        if dist[j][1] == 2:
            Fam_l+=1
    #print('bc_l  ', bc_l, '  cont  ', Cont_l, '  Fam  ', Fam_l)
    max_l = 0
    if bc_l > Cont_l:
        if bc_l > Fam_l:
            max = 0
        else: max = 2
    else:
        if Cont_l> Fam_l:
            max = 1
        else: max = 2
    if max == 0:
        New_BC += 1
    else:
        BC_error +=1
        All_error +=1

print(New_BC, '  ', BC_error)

###############

for i in range(data_Control.shape[0]):
    dist  = np.zeros((3*numbers, 2))
    for j in range(numbers):
        dist[3*j][0] = abs(data_Control[i] - train_data[0][j])
        dist[3*j][1] = 0
        dist[3*j+1][0] = abs(data_Control[i] - train_data[1][j])
        dist[3*j+1][1] = 1
        dist[3*j+2][0] = abs(data_Control[i] - train_data[2][j])
        dist[3*j+2][1] = 2
    
    dist = dist[dist[:,0].argsort(kind='mergesort')]
    
    bc_l =0
    Cont_l =0
    Fam_l =0
    for j in range(k):
        if dist[j][1] == 0:
            bc_l+=1
        if dist[j][1] == 1:
            Cont_l+=1
        if dist[j][1] == 2:
            Fam_l+=1
    #print('bc_l  ', bc_l, '  cont  ', Cont_l, '  Fam  ', Fam_l)
    max_l = 0
    if bc_l > Cont_l:
        if bc_l > Fam_l:
            max = 0
        else: max = 2
    else:
        if Cont_l> Fam_l:
            max = 1
        else: max = 2
    if max == 1:
        New_Control += 1
    else:
        Control_error +=1
        All_error +=1

print(New_Control, '  ', Control_error)


##############
for i in range(data_Fam.shape[0]):
    dist  = np.zeros((3*numbers, 2))
    for j in range(numbers):
        dist[3*j][0] = abs(data_Fam[i] - train_data[0][j])
        dist[3*j][1] = 0
        dist[3*j+1][0] = abs(data_Fam[i] - train_data[1][j])
        dist[3*j+1][1] = 1
        dist[3*j+2][0] = abs(data_Fam[i] - train_data[2][j])
        dist[3*j+2][1] = 2
    
    dist = dist[dist[:,0].argsort(kind='mergesort')]
    
    bc_l =0
    Cont_l =0
    Fam_l =0
    for j in range(k):
        if dist[j][1] == 0:
            bc_l+=1
        if dist[j][1] == 1:
            Cont_l+=1
        if dist[j][1] == 2:
            Fam_l+=1
    #print('bc_l  ', bc_l, '  cont  ', Cont_l, '  Fam  ', Fam_l)
    max_l = 0
    if bc_l > Cont_l:
        if bc_l > Fam_l:
            max = 0
        else: max = 2
    else:
        if Cont_l> Fam_l:
            max = 1
        else: max = 2
    if max == 2:
        New_Fam += 1
    else:
        Fam_error +=1
        All_error +=1

print(New_Fam, '  ', Fam_error)
