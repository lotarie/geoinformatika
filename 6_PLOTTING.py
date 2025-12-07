import matplotlib.pyplot as plt



#size of point clouds
N_sizes = [4, 550, 903, 6727, 12527, 18084, 25737]

#time naive search
time_naive = [0.0004, 0.4746, 1.2063, 65.7948, 223.8692, 454.5914, 691.1885] 
#time voxelization
time_voxel = [0.0004, 0.2583, 0.4265, 17.3561, 44.1883, 44.6765, 220.3681] 
#time kdtree
time_kdtree = [0.0003, 0.0287, 0.0567, 0.5512, 1.1548, 1.9204, 2.8009] 

n_voxels = [1, 5, 10, 15, 25, 50, 75]
time_n_voxels = [88.8113, 87.1660, 88.5828, 89.1403, 94.3484, 98.9473, 108.1614]

#plotting
plt.figure(figsize=(10, 6))
plt.plot(N_sizes, time_naive, label='Naive search', marker='o')
plt.title('KNN Search Time Comparison')
plt.xlabel(' Size of point cloud (N)')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(N_sizes, time_kdtree, label='KD-tree', marker='^')
plt.title('KNN Search Time Comparison')
plt.xlabel(' Size of point cloud (N)')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(N_sizes, time_voxel, label='Voxelization', marker='x')
plt.title('KNN Search Time Comparison')
plt.xlabel(' Size of point cloud (N)')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(n_voxels, time_n_voxels, label='Voxelization', marker='o')
plt.title('KNN Search Time Comparison')
plt.xlabel('Number of voxels')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)
plt.show()