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
        


X, Y, Z = loadPoints ('minitest.txt')

#amount of points
n=len(X) 

#compute number of bins along one axis
n_bins = (n**(1/3))

#number of voxels per row/column/depth
n_r = (n_bins**(1/3))
n_xyz = (n_bins**(1/3))

#initialize spatial index
x_min, y_min, z_min, dx, dy, dz, bx, by, bz = init_Spat_Index(X,Y,Z, n_xyz)

#compute 3D spatial index   
jx, jy, jz = get_3D_index ((( dx+x_min+x_min)/2), ((dy+y_min+y_min)/2), ((dz+z_min+z_min)/2), dx,dy,dz,x_min,y_min,z_min, n_xyz)

#compute 1D index
index = get_1D_index(jx, jy, jz, n_xyz)

#body = drawPoints(X,Y,Z, 0.1)
voxel = drawVoxels(x_min, y_min, z_min, dx, dy, dz, 1)