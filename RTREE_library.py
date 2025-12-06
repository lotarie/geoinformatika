from rtree import index
from math import *


def loadPoints(file):
    #Load file
    X, Y, Z = [], [], [] 
    
    with open(file) as f:
        
        for line in f:
            x, y, z = line.split('\t')
            
            X.append(float(x))
            Y.append(float(y))
            Z.append(float(z))
            
    points = list(zip(X, Y, Z))
    return points, X, Y, Z

def euclid_distance(x1, y1, z1, x2, y2, z2): #points are lists 
   #euclid distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
   
    return sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

points, X, Y, Z = loadPoints('tree_18.txt')

p = index.Property()
p.dimension = 3

#create rtree
idx = index.Index(properties=p)

for i, (x, y, z) in enumerate(points):
    
    idx.insert(id=i, coordinates=(x, y, z, x, y, z))
    
reference_point_index = 0
reference_point = points[reference_point_index]

k_neighbors = 5

nearest_ids1 = list(idx.nearest(coordinates=reference_point, num_results=k_neighbors,objects='id')) 
nearest_ids = [item.id for item in nearest_ids1]


nn_dist_coord = []
for rank, neighbor_id in enumerate(nearest_ids):
    neighbor_point = points[neighbor_id]
    d = euclid_distance(neighbor_point[0], neighbor_point[1], neighbor_point[2], reference_point[0], reference_point[1], reference_point[2])
    if d == 0: 
        continue
    else:
        nn_dist_coord.append((d, neighbor_point))
        
print(nn_dist_coord)


    
