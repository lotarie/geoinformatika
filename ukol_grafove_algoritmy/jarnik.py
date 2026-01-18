from lines_to_graph import loadEdges, pointsToIDs, edgesToGraph
from numpy import unique
import matplotlib.pyplot as plt
import heapq
import random

def jarnik(num_nodes, G, start_node=0):

    mst_weight = 0
    mst_edge_ids = []
    mst_edges_data = [] 

    #priority queue 
    #start with the start node, weight 0, no previous node or edge
    pq = [(0, start_node, -1, -1)] 
    
    visited = set()

    while pq and len(visited) < num_nodes:
        weight, u, prev_node, edge_id = heapq.heappop(pq)

        #if already visited, skip
        if u in visited:
            continue

        #mark as visited
        visited.add(u)

        #add edge to mst (if not starting node)
        if prev_node != -1:
            mst_weight += weight
            mst_edge_ids.append(edge_id)
            mst_edges_data.append((prev_node, u))

        #explore neighbors
    
        for v, data in G[u].items():
            if v not in visited:
                heapq.heappush(pq, (data['weight'], v, u, data['id']))

    return mst_edge_ids, mst_weight, mst_edges_data





file = 'lines.csv'

#data loading 
PS, PE, W, IDs = loadEdges(file)
PSE = PS + PE
PSE = unique(PSE, axis=0).tolist()
D, ID_to_Coords = pointsToIDs(PSE)
num_nodes = len(D)

#graph construction
G = edgesToGraph(D, PS, PE, W, IDs)

# run jarniks algorithm
start_node = 0 
mst_ids, total_cost, mst_data = jarnik(num_nodes, G, start_node)


print(f"total cost: {total_cost}")
print(f"number of edges in MST: {len(mst_ids)}")
# print(f"IDs in mst : {mst_ids}")

#visualization
plt.figure(figsize=(12, 10))
plt.title(f"Jarnik's MTS : {total_cost:.2f}")


for i in range(len(PS)):
    plt.plot([PS[i][0], PE[i][0]], [PS[i][1], PE[i][1]], 
             color='lightgray', linewidth=0.5, zorder=1)

#vizualization of MST edges
for u, v in mst_data:
    start_coord = ID_to_Coords[u]
    end_coord = ID_to_Coords[v]
    plt.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]], 
             color='red', linewidth=2, zorder=2)

#vizualization of all nodes


plt.legend()
plt.axis('equal')
plt.show()