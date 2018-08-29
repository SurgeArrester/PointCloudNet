#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 18:07:03 2018

@author: cameron
"""

import PointCloud as pc

import numpy as np
from sklearn.cluster import KMeans

class kMeansClassifier():
    def __init__(self, pointcloud):
        self.pointcloud = pointcloud
        self.kmeans = KMeans(n_clusters = pointcloud.numMols, random_state = 0).fit(pointcloud.molecule.xyz_cartn)
        
        print(pointcloud.cartesianCoords)
           
importPath = '/home/cameron/Desktop/xyz/'
testFile = 'T2_3686_num_molGeom.xyz'

pointcl = pc.PointCloud(importPath + testFile, inputType = 'xyz')
clusters = kMeansClassifier(pointcl)

print(pointcl.reducedCloudCart)
print(clusters.kmeans.cluster_centers_)