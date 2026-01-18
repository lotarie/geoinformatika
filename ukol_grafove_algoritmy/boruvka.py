from lines_to_graph import loadEdges, pointsToIDs, edgesToGraph
import matplotlib.pyplot as plt

 
#Load  data
PS, PE, W, IDs = loadEdges('lines.csv')

#create a list of all unique points 
all_points = list(set(PS) | set(PE))

#build the graph
point_map, id_to_coords = pointsToIDs(all_points)
G = edgesToGraph(point_map, PS, PE, W, IDs)



#initialize Union-Find structure
#each node starts as its own parent
parent = {node: node for node in G}

def find(i):
    #finds the root of the component for node i (with path compression)
    if parent[i] != i:
        parent[i] = find(parent[i])
    return parent[i]

def union(i, j):
    #unions the components of i and j, returns true if a merge happened
    root_i = find(i)
    root_j = find(j)
    if root_i != root_j:
        parent[root_i] = root_j
        return True
    return False

mst_edges = []
mst_weight = 0.0
num_components = len(G)



while num_components > 1:
    #dictionary to store the cheapest edge for each component
    cheapest = {}

    #iterate over all nodes and their edges to find the cheapest edge 
    for u in G:
        for v, data in G[u].items():
            root_u = find(u)
            root_v = find(v)

            if root_u != root_v:
                w = data['weight']
                #check if this edge is the cheapest for component root_u
                if root_u not in cheapest or w < cheapest[root_u][0]:
                    cheapest[root_u] = (w, u, v, data['id'])

    #if no edges were found, the graph is disconnected
    if not cheapest:
        break

    #add the cheapest edges to the MST and merge components
    for w, u, v, edge_id in cheapest.values():
        if union(u, v):
            mst_edges.append(edge_id)
            mst_weight += w
            num_components -= 1


print(f"Total MST Weight: {mst_weight}")
print(f"Number of Edges in MST: {len(mst_edges)}")



mst_edge_ids = set()
num_components = len(G)

while num_components > 1:
        cheapest = {}
        for u in G:
            for v, data in G[u].items():
                root_u, root_v = find(u), find(v)
                if root_u != root_v:
                    if root_u not in cheapest or data['weight'] < cheapest[root_u][0]:
                        cheapest[root_u] = (data['weight'], u, v, data['id'])
        
        if not cheapest: break

        for w, u, v, eid in cheapest.values():
            if union(u, v):
                mst_edge_ids.add(eid)
                num_components -= 1


#visualization of the MST
plt.figure(figsize=(10, 10))

    # Iterate through all input lines and plot them
for i in range(len(IDs)):
        # Extract coordinates
        x_values = [PS[i][0], PE[i][0]]
        y_values = [PS[i][1], PE[i][1]]
        edge_id = IDs[i]

        if edge_id in mst_edges:
            plt.plot(x_values, y_values, color='red', linewidth=1.5, zorder=2)
        else:
            plt.plot(x_values, y_values, color='lightgray', linewidth=0.5, zorder=1)


plt.plot([], [], color='red', linewidth=1.5, label='MST Edges')
plt.plot([], [], color='lightgray', linewidth=0.5, label='Edges')

plt.title("Boruvka's MST Visualization")
plt.legend()
plt.axis('equal')
plt.show()