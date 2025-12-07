from math import *
from time import *
from matplotlib.pyplot import *
from numpy import * 
from collections import defaultdict
from sklearn.neighbors import KDTree
import datetime
from numpy.linalg import *


def loadPoints(file):
    #Load file
    X, Y, Z = [], [], [] 
    
    with open(file) as f:
        
        for line in f:
            x, y, z = line.split('\t')
            
            X.append(float(x))
            Y.append(float(y))
            Z.append(float(z))
    
    return X, Y, Z


def euclid_distance(x1, y1, z1, x2, y2, z2): #points are lists 
   #euclid distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
   
    return sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

def naive_search (X, Y, Z, x, y, z, k):
    distances = []
    
    for i in range(len(X)): 
        px, py, pz = X[i], Y[i], Z[i]
        compute_distance = euclid_distance(px, py, pz, x, y, z)
        if compute_distance ==0:
            continue
        distances.append((compute_distance, i)) #append distance and point index 
        
    distances.sort() 
    
    
    Xn, Yn, Zn = [], [], []  
    
    for _, i in distances[:k]:
        Xn.append(X[i])  #add point to the list 
        Yn.append(Y[i])
        Zn.append(Z[i])
    
    return Xn, Yn, Zn

start_time = time.perf_counter()

def compute_density(X, Y, Z, knn_method):
    distances = []

    for i in range(len(X)):
        #calculate nearest neighbour and its distance
        xn, yn, zn = knn_method(X, Y, Z, X[i], Y[i], Z[i], k=1) 
        d = euclid_distance(X[i], Y[i], Z[i], xn[0], yn[0], zn[0])
        distances.append(d)
    
    #calculate average distance
    daver = sum(distances)/len(distances)  

    rho = 1/daver**3

    return rho

#test point clouds
#X, Y, Z = loadPoints ('minitest.txt')
X, Y, Z = loadPoints('tree_18.txt')
#X, Y, Z = loadPoints ('test1000.txt')
#X, Y, Z = loadPoints('test5000.txt')
#X, Y, Z = loadPoints('test550.txt')
#X, Y, Z = loadPoints('test_half.txt')
#X, Y, Z  = loadPoints('test18000.txt')


#compute density
densnaive = compute_density(X, Y, Z, naive_search)
print(densnaive)




end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(elapsed_time)