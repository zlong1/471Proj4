# Zach Long
# CMSC 471
# Proj 4
# Takes 2 args, number of clusters and file.txt where file.txt is a series of 2d points, one per line.
# Displays a graph of points divided into their clusters using K-Means clustering or Lloyd's Algorithm
# Note that this algorithm is prone to falling into local minima and is not guaranteed to be correct.

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys

# Associates points with their nearest cluster center
def getAssociations(points, clusterCenters):
    associations = []
    for point in points:
        distance = np.linalg.norm(point - clusterCenters[0])
        cluster = 0
        for i in range(1, len(clusterCenters)):
            newDistance = np.linalg.norm(point - clusterCenters[i])
            if newDistance < distance:
                distance = newDistance
                cluster = i
        associations.append(cluster)
    return associations

# Returns centroids of a cluster of points
def getClusterCentroids(points, clusterCenters, associations):
    counts = []
    masses = []
    for i in range(len(clusterCenters)):
        counts.append(0)
        masses.append([0,0])
    for i in range(len(points)):
        masses[associations[i]][0] += points[i][0]
        masses[associations[i]][1] += points[i][1]
        counts[associations[i]] += 1
    for i in range(len(clusterCenters)):
        clusterCenters[i] = [(masses[i][0] / max(counts[i],1)), (masses[i][1] / max(counts[i],1))]
    return clusterCenters

# Performs KMeans Clustering on an np array of points. Returns an array labeling each point by cluster ie [0 1 2 0 1]
def kMeans(points, numberOfClusters):
    # Step 1: Generate random start points
    # clusterCenters = random.sample(points, numberOfClusters)
    clusterCenters = []
    xMax = max(points[:,0])
    xMin = min(points[:,0])
    yMax = max(points[:,1])
    yMin = min(points[:,1])
    for i in range(numberOfClusters):
        clusterCenters.append([random.uniform(xMin, xMax),random.uniform(yMin, yMax)])
    # Above picks random points inside domain and range of points as basis for initial cluster centers.
    # Will likely be better if our clusters are not even in size. Alternatively can use commented out line
    # instead to simply pick random points in our data set to start with

    stop = False
    while not stop:
        oldCenters = list(clusterCenters)
        # Step 2: Associate each point with a cluster, decided by shortest distance to each center
        associations = getAssociations(points, clusterCenters)
        # Step 3: Set each cluster center as center mass of cluster
        clusterCenters = getClusterCentroids(points, clusterCenters, associations)
        # Repeat steps 2/3 until convergence
        if np.array_equal(oldCenters,clusterCenters):
            stop = True
    return associations

def main():
    points = []
    infile = open(sys.argv[2], "r")
    for line in infile:
        rawData = line.split()
        points.append([float(rawData[0]), float(rawData[1])])
    points = np.array(points)
    numberOfClusters = int(sys.argv[1])
    colorings = kMeans(points, numberOfClusters)
    # colorings = KMeans(n_clusters=numberOfClusters).fit_predict(points)
    plt.scatter(points[:,0], points[:,1], c=colorings)
    plt.show()

main()