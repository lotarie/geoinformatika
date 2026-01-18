from lines_to_graph import loadEdges, pointsToIDs, edgesToGraph
import heapq
from collections import defaultdict
from numpy import unique
import matplotlib.pyplot as plt

def dijkstra(G, start_node, end_node):
    # distances: dictionary node to distance
    queue = [(0, start_node)]
    distances = {node: float('inf') for node in G}
    distances[start_node] = 0
    predecessors = {node: None for node in G}# predecessors: node -> previous node with the shortest distance
    
    while queue:
        current_dist, u = heapq.heappop(queue)
        if current_dist > distances[u]:
            continue
        
        if u == end_node:
            break

        for v, edge_data in G[u].items():
            weight = edge_data['weight'] 
            new_dist = current_dist + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                predecessors[v] = u
                heapq.heappush(queue, (new_dist, v))
    # path reconstruction           
    path_nodes = []
    path_edge_ids = [] # ids on the path
    current = end_node
    
    if distances[end_node] == float('inf'):
        return [], [], float('inf')
        
    while current is not None:
        path_nodes.append(current)
        prev = predecessors[current]
        if prev is not None:
            # ID edge, which leads from previous to current node
            edge_id = G[prev][current]['id']
            path_edge_ids.append(edge_id)
        current = prev
            
    return path_nodes[::-1], path_edge_ids[::-1], distances[end_node]

## Using dijkstra for search of the shortest path in the example 'lines.csv'
file = 'lines.csv'

# Loading edges in the file
PS, PE, W, IDs= loadEdges(file)
PSE = PS + PE
PSE= unique(PSE,axis=0).tolist() #Merge lists and remove unique points
D, _ = pointsToIDs(PSE)

# creating the non-oriented graph
G = edgesToGraph(D, PS, PE, W, IDs)

#0-343
start_node = 70
end_node = 324
path_nodes, path_edges, cost = dijkstra(G, start_node, end_node)


print(f"The shortest path found. The distance: {cost}")
print(f"IDs of the path edges: {path_edges}")

# Visualization
plt.figure(figsize=(10, 8))

for i in range(len(PS)): # visualisation of the graph with all nodes
    x_values = [PS[i][0], PE[i][0]]
    y_values = [PS[i][1], PE[i][1]]
    plt.plot(x_values, y_values, color='lightgray', zorder=1)
    
for i in range(len(path_nodes)-1): # visualisation of the reconstructed path
    u = path_nodes[i]
    v = path_nodes[i+1]
    x_values = [PSE[u][0], PSE[v][0]]
    y_values = [PSE[u][1], PSE[v][1]]
    plt.plot(x_values, y_values, color='red', linewidth=2, zorder=2)

plt.scatter(*zip(*PSE), color='blue', s=5, zorder=3)
plt.title('Shortest path using dijkstra algoritm')
plt.xlabel('X coordinates')
plt.ylabel('Y coordinates')
plt.grid()
plt.show()
