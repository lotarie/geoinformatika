import numpy as np
import open3d as o3d 
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

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

def loadfile (file):
    curvatures = []
    
    with open(file) as f: 
        for line in f:
            curvatures.append(float(line.strip()))
    return np.array(curvatures)


X, Y, Z = loadPoints('tree_18.txt') #25675
curvatures = loadfile('curvature_kdtree.txt')

X_list = [X]
Y_list = [Y]
Z_list = [Z]


#convert curvatures to a NumPy array
curvatures = np.array([curvatures]) 

points = np.stack((X, Y, Z), axis=1) 

#normalization 
min_kappa = np.min(curvatures)
max_kappa = np.max(curvatures)

#if min and max are the same, avoid division by zero
if max_kappa == min_kappa:
    normalized_curvatures = np.zeros_like(curvatures)
else:
    normalized_curvatures = (curvatures - min_kappa) / (max_kappa - min_kappa)

#define colormap
cmap = cm.get_cmap('jet') 

colors_rgba = cmap(normalized_curvatures)
colors_rgb = colors_rgba[:, :3]

#create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([pcd])



fig = plt.figure(figsize=(2, 8)) 

cbar_ax = fig.add_axes([0.4, 0.05, 0.2, 0.9]) 

norm = mcolors.Normalize(vmin=min_kappa, vmax=max_kappa)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]) 
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

cbar.set_label('curvature', rotation=270, labelpad=20)
cbar.ax.set_title('color scale', loc='left')

plt.show()