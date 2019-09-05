import numpy as np
import os
import math
import matplotlib.pyplot as plt


def khaciyan_algorithm (P, toleranse):
    dimension = P.shape[0]
    N = P.shape[1]

    count = 1
    err = 1

    Q = np.vstack((P,np.ones(N)))
    print(Q)

    u = np.ones(N)/N
    print(u)

    while err > toleranse:
        
        U = np.diag(u)
        #print(Q.shape, U.shape, np.transpose(Q).shape)
        X = np.dot(np.dot(Q, U), np.transpose(Q))
        

        M = np.diag(np.dot(np.dot(np.transpose(Q), np.linalg.inv(X)), Q))
        
        maximum = np.max(M)
        j = np.argmax(M)
        

        step_size = (maximum - dimension - 1)/((dimension +1)*(maximum -1))

        print('sssss ',step_size)
        new_u = (1- step_size)*u
        new_u[j] = new_u[j] + step_size
        
        #err = (maximum - dimension - 1)/(dimension +1)
        err = np.linalg.norm(new_u - u)
        print(err)
        print("##", np.linalg.det(X))
        count +=1
        u = new_u
    
    #print(X)
    print(u)

    #B = X[0:dimension,0:dimension]
    #b = X[-1,0:dimension]
    #print(b)
    #c = -1* np.dot(np.linalg.inv(B),b)
    #A = B/ (1 + np.dot(np.dot(np.linalg.inv(B),b),b) - B[-1,-1])
    
    U = np.diag(u)

    pup  = np.dot(np.dot(P, U), np.transpose(P))
    pu = np.dot(P,u)
    pu_1 = pu.reshape((1,dimension))
    pu_2 = pu.reshape((dimension,1))
    pupu = np.dot(pu_2,pu_1)
    pup_pupu = pup - pupu
    A = np.linalg.inv(pup_pupu)/dimension

    c =  np.dot(P,u)
    print('##',pupu)
    #print(X)
    print(np.linalg.eig(A))
    return A,c


def print_elipse (P,G,c):
    # G - это A из алгоритма
    # L = Ax**2 + Bxy + Cy**2 + Dx + Ey + F = 0
    A = G[0][0]
    B = G[1][0]  + G[0][1]
    C = G[1][1]
    D = -2*G[0][0]*c[0] - G[1][0]*c[1] - G[0][1]*c[1]
    E = -G[1][0]*c[0] - G[0][1]*c[0] -2*G[1][1]*c[1]
    F = G[0][0]*c[0]**2 +G[1][0]*c[0]*c[1] + G[0][1]*c[0]*c[1] + G[1][1]*c[1]**2 -1

    x = np.linspace(np.min(P[0])-1, np.max(P[0])+1, 1000)
    y = np.linspace(np.min(P[1])-1, np.max(P[1])+1, 1000)

    # for i in range(100):
    #     for j in range(100):
    #         test = np.array([x[i],y[j]])
    #         if 0< np.dot(np.dot((test - c), G), (test - c)) <= 1:
    #             plt.scatter(x[i],y[j], s = 2, c = 'black')

    for i in range(P.shape[1]):
        plt.scatter(P[0][i], P[1][i])
    
    
    x,y= np.meshgrid(x,y)

    L = A*x**2 + B*x*y + C*y**2 + D*x + E*y + F

    test = np.array([P[0][0], P[1][0]])
    #test = np.array([1.9, 3.135])
    #test = np.array([0,3])
    print(np.dot(np.dot((test - c), G), (test - c)))
    print(A*test[0]**2+ B*test[0]*test[1] + C*test[1]**2+ D*test[0] + E*test[1] + F)

    
    plt.contour(x,y,L,[0])

    plt.xlim([np.min(P[0])-1, np.max(P[0])+1])
    plt.ylim([np.min(P[1])-1, np.max(P[1])+1])

    plt.show()




if __name__ == '__main__':
    P = np.array([[1,1,2,3],[1,2,2,4]])
    print(P.shape)
    print(P)
    A, c = khaciyan_algorithm(P, 10**-5)
    print(A, c)
    print_elipse(P,A,c)