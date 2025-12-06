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


def curvature(X, Y, Z, knn_method):
    kappa = []
    #compute curvature for each point
    for i in range(len(X)):
        #get k nearest neighbours
        xn, yn, zn = knn_method(X, Y, Z, X[i], Y[i], Z[i], k=30) 
        
        #compute covariance matrix
        points = array(list(zip(xn, yn, zn)))
        centroid = points.mean(axis=0)
        centered_points = points - centroid  
        cov_matrix = centered_points.T @ centered_points / len(points)
        
        #compute eigenvalues
        eigenvalues, _ = eig(cov_matrix)
        eigenvalues = sorted(eigenvalues)
        
        #compute curvature - kappa = lambda1 / (lambda1 + lambda2 + lambda3)
        if sum(eigenvalues) == 0:
            kappa.append(0)
        else:
            kappa.append(eigenvalues[0] / sum(eigenvalues))
            
    return kappa


class Node:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right
        
def build_kdtree(points, depth=0):

    if not points:
        return None
    
    kd = 3  #3D points
    axis = depth % kd
    
    #sort points by the selected axis and choose median as pivot element
    points.sort(key=lambda point: point[axis])
    #compute median index
    median = len(points) // 2
    
    #create node and construct subtrees
    return Node(
        point=points[median],
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1)
    )

 #function to add a point to the knn list
def knn_list_add(knn_list, point, dist, k):
    if dist == 0:
        return

    if len(knn_list) < k:
        knn_list.append((dist, point))
    else:
        #find the farthest neighbor
        max_i = 0
        for i in range(1, len(knn_list)):
            if knn_list[i][0] > knn_list[max_i][0]:
                max_i = i

        #replace if the new point is closer
        if dist < knn_list[max_i][0]:
            knn_list[max_i] = (dist, point)

    #keep the list sorted
    knn_list.sort()


# KD-Tree k-NN search
def kd_tree_knn_search(node, target, k, knn_list, depth=0):
    if node is None:
        return

    axis = depth % 3

    px, py, pz = node.point
    tx, ty, tz = target
    
    # compute distance to the current node
    dist = euclid_distance(tx, ty, tz, px, py, pz)
    knn_list_add(knn_list, node.point, dist, k)

    #choose which branch to explore first
    if target[axis] < node.point[axis]:
        near = node.left
        far = node.right
    else:
        near = node.right
        far = node.left

    # explore the near branch
    kd_tree_knn_search(near, target, k, knn_list, depth + 1)

    # explore the far branch if needed
    if len(knn_list) < k or abs(target[axis] - node.point[axis]) < knn_list[-1][0]:
        kd_tree_knn_search(far, target, k, knn_list, depth + 1)


#wrapper function for KD-Tree k-NN search
def kd_tree_search_wrapper(X, Y, Z, x, y, z, k):
    target = (x, y, z)
    knn_list = []
    kd_tree_knn_search(kdtree_root, target, k, knn_list)
    
    Xn, Yn, Zn = [], [], []
    for _, p in knn_list:
        Xn.append(p[0])
        Yn.append(p[1])
        Zn.append(p[2])

    return Xn, Yn, Zn

#X, Y, Z = loadPoints ('minitest.txt')
X, Y, Z = loadPoints('tree_18.txt')
#X, Y, Z = loadPoints ('test1000.txt')
#X, Y, Z = loadPoints('test5000.txt')

print(len(X))

kdtree_root = build_kdtree(list(zip(X, Y, Z)))

tset =kd_tree_search_wrapper(X, Y, Z, X[0], Y[0], Z[0], k=5)
print(tset)

#dens = compute_density(X, Y, Z, kd_tree_search_wrapper)
#print(dens)

#naivenn = naive_search(X, Y, Z, X[0], Y[0], Z[0], k=5)
#print(naivenn)


curvatur = curvature(X, Y, Z, kd_tree_search_wrapper)
print(curvatur)

import numpy as np

# Příklad NumPy pole
results = np.array(curvatur)

# Název souboru, do kterého chceme data uložit
file_name = 'curvature_kdtree.txt'

# Uložení pole do textového souboru
np.savetxt(
    file_name, 
    results, 
)

print(f"NumPy pole bylo úspěšně uloženo do souboru: {file_name}")

#densnaive = compute_density(X, Y, Z, naive_search)
#print(densnaive)



