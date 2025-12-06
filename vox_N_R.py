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

 #Draw voxels  
def drawVoxels(x_min, y_min, z_min, dx, dy, dz, V): #V = 3Dnumpyarray
    # Create figure
    fig = figure()
    ax = axes(projection = '3d')
    ax.set_aspect('equal') 
    

    
    #Create meshgrid
    xedges = linspace((x_min), (x_min) + (dx), (n_r+1))
    yedges = linspace((y_min), (y_min) + (dy), (n_r+1))
    zedges = linspace((z_min), (z_min) + (dz), (n_r+1))
    
    
    VX, VY, VZ = meshgrid(xedges, yedges, zedges,  indexing="ij") 
    
    #Draw voxels
    ax.voxels(VX, VY, VZ, V, edgecolor='k')
    
    show()
 
def init_Spat_Index(X,Y,Z, n_r):
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
    bx = dx/n_r
    by = dy/n_r
    bz = dz/n_r
    
    return x_min, y_min, z_min, dx, dy, dz, bx, by, bz


#compute spatial index
def get_3D_index (x, y, z, dx,dy,dz,x_min,y_min,z_min, n_r):
    c=0.99999
    #reduce coordinates
    xr = ((x-x_min)/dx)   
    yr = ((y-y_min)/dy)
    zr = ((z-z_min)/dz)
    
    #compute spatial indices 
    
    jx = int(xr * c * n_r)
    jy = int(yr * c * n_r)
    jz = int(zr * c * n_r)
    
    return jx, jy, jz
 
def get_1D_index(jx, jy, jz, n_r):  
      
    return jx + jy * n_r + jz * n_r**2

def create3Dindex(X,Y,Z, x_min, y_min, z_min, dx, dy, dz, bx, by, bz, n_r):
    H = defaultdict(list)   #for 1 key multiple values
    J = [0] * len(X) 
    
    for i in range(len(X)): 
        jx, jy, jz = get_3D_index (X[i], Y[i], Z[i], dx, dy, dz, x_min, y_min, z_min, n_r)
        idx = get_1D_index (jx, jy, jz, n_r)
        H[idx].append(i)  #store point index i for each voxel index
        J[i] = idx
        
    return H, J


def euclid_distance(x1, y1, z1, x2, y2, z2): #points are lists 

   #euclid distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
   
    return sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)



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


#X, Y, Z = loadPoints ('minitest.txt')
X, Y, Z = loadPoints('tree_18.txt')
#X, Y, Z = loadPoints ('test1000.txt')
#X, Y, Z = loadPoints('test5000.txt')
#X, Y, Z = loadPoints('test550.txt')
#X, Y, Z = loadPoints('test_half.txt')
#X, Y, Z  = loadPoints('test18000.txt')

#amount of points
n=len(X) 
#compute number of bins along one axis
n_bins = int(n**(1/3))



#number of voxels per row/column
n_r = 1

#initialize spatial index
x_min, y_min, z_min, dx, dy, dz, bx, by, bz = init_Spat_Index(X,Y,Z, n_r)

#compute 3D spatial index   
jx, jy, jz = get_3D_index ((( dx+x_min+x_min)/2), ((dy+y_min+y_min)/2), ((dz+z_min+z_min)/2), dx,dy,dz,x_min,y_min,z_min, n_r)

#compute 1D index
index = get_1D_index(jx, jy, jz, n_r)

#create 3D spatial index
H, J = create3Dindex(X,Y,Z, x_min, y_min, z_min, dx, dy, dz, bx, by, bz, n_r)
print(J)


def knn_search_voxel(X, Y, Z, x, y, z, k):
    distances = []
    
    points_in_voxel = H[index]
    
    for i in points_in_voxel: 
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


#body = drawPoints(X,Y,Z, 0.025)
#Create voxel array
V = zeros((n_r, n_r, n_r), dtype=bool)

for i in range(len(X)):
    jx, jy, jz = get_3D_index (X[i], Y[i], Z[i], dx,dy,dz,x_min,y_min,z_min, n_r)
    V[jx, jy, jz] = True

#voxel = drawVoxels(x_min, y_min, z_min, dx, dy, dz, V)

#knn = knn_search_voxel(X, Y, Z, X[0], Y[0], Z[0], k=2)
#print(knn)

#curvatu = curvature(X, Y, Z, knn_search_voxel)
#print(curvature)

#den = compute_density(X, Y, Z, knn_search_voxel)
#print(den)

end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(elapsed_time)
