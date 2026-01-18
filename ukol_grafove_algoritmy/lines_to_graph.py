from collections import *
from numpy import *

#Load edges
def loadEdges(file_name):
    #Convert list of lines to the graph
    PS = []
    PE = []
    W = []
    IDs = [] 
    
    with open(file_name, encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            parts = line.split()
            if not parts: continue
            x1, y1, x2, y2, w, ids= parts
            
            PS.append((float(x1), float(y1)))
            PE.append((float(x2), float(y2)))
            W.append(float(w))
            IDs.append(int(ids)) 
            
    return PS, PE, W, IDs


def pointsToIDs(P):
    D = {}
    ID_to_Coords = {}
    for i in range(len(P)):
        coord = (P[i][0], P[i][1])
        D[coord] = i
        ID_to_Coords[i] = coord
    return D, ID_to_Coords

def edgesToGraph(D, PS, PE, W, IDs):
    G = defaultdict(dict)

    for i in range(len(PS)):
        u = D[PS[i]]
        v = D[PE[i]]
        w = W[i]
        edge_id = IDs[i] 
        edge_data = {'weight': w, 'id': edge_id}
        
        G[u][v] = edge_data
        G[v][u] = edge_data 

    return G