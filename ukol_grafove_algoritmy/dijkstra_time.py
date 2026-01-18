
from collections import defaultdict
from numpy import unique
import heapq
from numpy import unique
import matplotlib.pyplot as plt


def dijkstra(G, start_node, end_node):
    queue = [(0, start_node)]
    travel_times = {node: float('inf') for node in G}
    travel_times[start_node] = 0
    predecessors = {node: None for node in G}# predecessors: node -> previous node with the shortest time
    
    while queue:
        current_time, u = heapq.heappop(queue)
        if current_time > travel_times[u]:
            continue
        
        if u == end_node:
            break

        for v, edge_data in G[u].items():
            travel_time = edge_data['travel time [mins]']
            new_time = current_time + travel_time
            if new_time < travel_times[v]:
                travel_times[v] = new_time
                predecessors[v] = u
                heapq.heappush(queue, (new_time, v))
                
    # path reconstruction           
    path_nodes = []
    path_edge_ids = [] # ids on the path
    current = end_node
    if travel_times[end_node] == float('inf'):
        return [], [], float('inf')
    
    while current is not None:
        path_nodes.append(current)
        prev = predecessors[current]
        if prev is not None:
            # ID edge, which leads from previous to current node
            edge_id = G[prev][current]['id']
            path_edge_ids.append(edge_id)
        current = prev
            
    return path_nodes[::-1], path_edge_ids[::-1], travel_times[end_node]

#Load edges
def loadEdges(file_name):     #Convert list of lines to the graph
    PS = []
    PE = []
    L = []
    IDs = []
    C = [] # class of communication - maximal velocity class
    with open(file_name, encoding='utf-8-sig') as f:
        for line in f:
            parts = line.split()
            #Add start, end points and weights to the list
            c, l, x1, y1, x2, y2, ids= parts
            
            PS.append((float(x1), float(y1)))
            PE.append((float(x2), float(y2)))
            L.append(float(l))
            IDs.append(int(ids)) 
            C.append(int(c))
            
    return PS, PE, L, C, IDs


def pointsToIDs(P): #Create a map: key = coordinates, value = id
    D = {}
    ID_to_Coords = {}
    for i in range(len(P)):
        coord = (P[i][0], P[i][1])
        D[coord] = i
        ID_to_Coords[i] = coord
    return D, ID_to_Coords

def edgesTravelTimeToGraph(L, PS, PE, C, IDs):
    GT = defaultdict(dict)
    for i in range(len(PS)):
        u = D[PS[i]]
        v = D[PE[i]]
        l = L[i]
        c = C[i]
        if c == 6:
            vel = 30
        if c == 5:
            vel = 40
        if c == 4:
            vel = 60
        if c == 3:
           vel = 90
        travel_time_mins = ((l/1000)/vel*60)
        edge_id = IDs[i]
        length_data = {'travel time [mins]': travel_time_mins,'id': edge_id}
        GT[u][v] = length_data
        GT[v][u] = length_data 

    return GT

## Example of using dijkstra to find the shortest travel time 'id_classes.txt'
file = 'id_classes.txt'

# Loading edges in the file
PS, PE, L, C, IDs= loadEdges(file)
PSE = PS + PE
PSE= unique(PSE,axis=0).tolist() #Merge lists and remove unique points
D, _ = pointsToIDs(PSE)
GT= edgesTravelTimeToGraph(L, PS, PE, C, IDs)
print(GT)

start_node = 70
end_node = 324

path_nodes, path_edges, times = dijkstra(GT, start_node, end_node)


print(f"The shortest path found. The time: {times}")
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
