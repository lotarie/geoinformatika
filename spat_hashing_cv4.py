from math import *
from time import *
from matplotlib.pyplot import *
from numpy import * 
from collections import defaultdict
from sklearn.neighbors import KDTree
import datetime
from numpy.linalg import *
import time



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




def getNN(xq, yq, zq, X, Y, Z):
    #Find nearest point and its distance
    dmin = inf
    xn, yn, zn = X[0], Y[0], Z[0]
    
    for i in range(len(X)):
        #Compute distance
        dx, dy, dz = xq - X[i], yq - Y[i], zq - Z[i]
        d = (dx*dx + dy*dy + dz * dz)**0.5
        
        #Actualize minimum: distance + coordinates
        if d < dmin and d > 0:
            dmin = d
            xn, yn, zn = X[i], Y[i], Z[i]
    return xn, yn, zn, dmin


def drawPoints(X, Y, Z, bx, transp = 0.2):
    # Create figure
    fig = figure()
    ax = axes(projection = '3d')
    ax.set_aspect('equal')

    #Compute sphere scale: 1 pix = 25.4 mm
    scale = 1
    if bx > 0:
        scale = int(bx * bx * 40 * 40)
        
    #Plot points
    ax.scatter(X, Y, Z, s=scale, alpha = transp)

    show()

    
def drawVoxels(x_min, y_min, z_min, dx, dy, dz, V):
    # Create figure
    fig = figure()
    ax = axes(projection = '3d')
    ax.set_aspect('equal') 
    
    #Create meshgrid
    xedges = linspace(x_min, x_min + dx, n_r+1)
    yedges = linspace(y_min, y_min + dy, n_r+1)
    zedges = linspace(z_min, z_min + dz, n_r+1)
    
    VX, VY, VZ = meshgrid(xedges, yedges, zedges,  indexing="ij") 
    
    #Draw voxels
    ax.voxels(VX, VY, VZ, V, edgecolor='k')
    
    show()
 
def init_Spat_Index(X,Y,Z, n_xyz):
    x_min=min(X)
    x_max=max(X)
    y_min=min(Y)
    y_max=max(Y)
    z_min=min(Z)
    z_max=max(Z)
    
     #minmax box edges
    dx = (x_max-x_min)
    dy = (y_max-y_min)
    dz = (z_max-z_min)
    
    #size of grid cell
    bx = dx/n_xyz
    by = dy/n_xyz
    bz = dz/n_xyz
    
    return x_min, y_min, z_min, dx, dy, dz, bx, by, bz


#compute spatial index
def get_3D_index (x, y, z, dx,dy,dz,x_min,y_min,z_min, n_xyz):
    c=0.99999
    #reduce coordinates
    xr = ((x-x_min)/dx)   
    yr = ((y-y_min)/dy)
    zr = ((z-z_min)/dz)
    
    #compute spatial indices 
    
    jx = int(xr * c * n_xyz)
    jy = int(yr * c * n_xyz)
    jz = int(zr * c * n_xyz)
    
    return jx, jy, jz
 
def get_1D_index(jx, jy, jz, n_xyz):  
      
    return jx + jy * n_xyz + jz * n_xyz**2

def create3Dindex(X,Y,Z, x_min, y_min, z_min, dx, dy, dz, bx, by, bz, n_xyz):
    H = defaultdict()   #for 1 key multiple values
    
    for i in range(len(X)): 
        jx, jy, jz = get_3D_index (X[i], Y[i], Z[i], dx,dy,dz,x_min,y_min,z_min, n_xyz)
        idx = get_1D_index (jx, jy, jz, n_xyz)
        H[idx].append(idx)
    
def euclid_distance(x1, y1, z1, x2, y2, z2): #points are lists 
   #euclid distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
   
    return sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


def naive_search (X, Y, Z, x, y, z, k):
    distances = []
    
    for i in range(len(X)): 
        px, py, pz = X[i], Y[i], Z[i]
        compute_distance = euclid_distance(px, py, pz, x, y, z)
        distances.append((compute_distance, i)) #append distance and point index 
        
    distances.sort() 
    distances = distances[1:]  #first distance is distance to myself which is zero
    
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


#curvature

def curvature():
    pass


   
#Load points
#X, Y, Z = loadPoints('s:/K155/Public/155YGEI/cv12/tree_18.txt')
#X, Y, Z = loadPoints('tree_18.txt')


#test cloudpoint
X, Y, Z = loadPoints ('minitest.txt')

#amount of points
n=len(X) 


#amount of bins/voxels per x,y,z
n_bins = (n**(1/3))
 
n_xyz = (n_bins**(1/3))

#init. spatial index
x_min, y_min, z_min, dx, dy, dz, bx, by, bz = init_Spat_Index(X,Y,Z, n_xyz)
    
jx, jy, jz = get_3D_index ((( dx+x_min+x_min)/2), ((dy+y_min+y_min)/2), ((dz+z_min+z_min)/2), dx,dy,dz,x_min,y_min,z_min, n_xyz)

index = get_1D_index(jx, jy, jz, n_xyz)


nn_naive = naive_search(X,Y,Z, X[1], Y[1], Z[1], 3)
print(nn_naive)

dens = compute_density(X, Y, Z, naive_search) #takes about 15 mins
print(dens)

