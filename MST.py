# -*- coding: utf-8 -*-

import diffpy.Structure
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance

cifFilePath = '/home/cameron/Dropbox/T2_Dataset/molGeom/T2_2_num_molGeom.cif'

def loadFile(filePath):
    molecule = diffpy.Structure.loadStructure(filePath)
    coords = molecule.xyz_cartn
    np_coords = np.array(coords)    # Simple cartesian coords
    return np_coords

def adjacencyMatrixNN(coords, knn = 10):    
    dist = distance.pdist(coords, 'euclidean') # Create condensed distance matrix
    adjacency = distance.squareform(dist) # Create standard adjacency matrix
    for i, row in enumerate(adjacency):
        lowestVals = np.partition(row, knn-1)[:knn] # take k neighbours
        threshold = lowestVals.max()    # Take the longest distance from k neighbours
        exceedsThresholdFlags = row > threshold
        adjacency[i][exceedsThresholdFlags] = 0
    return adjacency

def networkPlot3D(G, angle, coords):
    # 3D network plot
    with plt.style.context(('ggplot')):       
        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)       
        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for value in coords:
            xi = value[0]
            yi = value[1]
            zi = value[2]           
            # Scatter plot
            ax.scatter(xi, yi, zi, c='blue',edgecolors='k', alpha=0.7)       
        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i,j in enumerate(G.edges()):
            x = np.array((coords[j[0]][0], coords[j[1]][0]))
            y = np.array((coords[j[0]][1], coords[j[1]][1]))
            z = np.array((coords[j[0]][2], coords[j[1]][2]))
        
        # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)
    # Set the initial view
    ax.view_init(30, angle) 
    # Hide the axes
    plt.show()   
    return

start_time = time.time()

coords = loadFile(cifFilePath)
adjacency = adjacencyMatrixNN(coords, 10)

graph = nx.to_networkx_graph(adjacency) # initialise graph
mst = nx.minimum_spanning_tree(graph)

networkPlot3D(mst, 20, coords)

print("Running time %s seconds" % (time.time() - start_time))
