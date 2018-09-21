#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 18:07:03 2018

@author: cameron
"""
from __future__ import division
import os
import time 

import PointCloud as pc

from sklearn.cluster import KMeans

directory = '/home/cameron/Desktop/xyz/coords/'

class kMeansClassifier():
    def __init__(self, pc):
        self.pc = pc
        self.kmeans = KMeans(n_clusters=pc.numMols, # using scipy implementation of kmeans
                             random_state=0).fit(pc.cartesianCoords)
        self.target = pc.generateTarget(pc.cartesianCoords)   # generate targets
        self.classifications = self.kmeans.predict(pc.cartesianCoords) # predict clusters
        
        # Run performance metrics
        self.contingencyTable = self.generateContingencyTable(self.target, self.classifications)
        self.accuracy = self.calcAccuracy(self.contingencyTable)
        self.precision = self.calcPrecision(self.contingencyTable)
        self.recall = self.calcRecall(self.contingencyTable)
        self.falsePositiveRate = self.calcFPRate(self.contingencyTable)
        self.fMeasure = self.calcFMeasure(self.contingencyTable)
        
    def generateContingencyTable(self, labels, classifications):
        '''
        Loop through each pair of points and evaluate whether they have been 
        placed into the correct clusters and return the counts as a contingency
        table [truePositive, falsePositive, falseNegative, trueNegative]
        '''
        truePositive = falsePositive = trueNegative = falseNegative = 0

        for i in range(len(labels)):
            for j in range(i, len(labels)):
                if labels[i] == labels[j] and classifications[i] == classifications[j]:
                    truePositive += 1
                elif labels[i] != labels[j] and classifications[i] == classifications[j]:
                    falsePositive += 1
                elif labels[i] == labels[j] and classifications[i] != classifications[j]:
                    falseNegative += 1
                elif labels[i] != labels[j] and classifications[i] != classifications[j]:
                    trueNegative += 1
        return [truePositive, falsePositive, falseNegative, trueNegative]
    
    def calcAccuracy(self, contingencyTable):
        return (contingencyTable[0] + contingencyTable[3]) / (sum(contingencyTable))
    
    def calcPrecision(self, contingencyTable):
        return contingencyTable[0] / (contingencyTable[0] + contingencyTable[1])
    
    def calcRecall(self, contingencyTable):
        return contingencyTable[0] / (contingencyTable[0] + contingencyTable[2])
    
    def calcFPRate(self, contingencyTable):
        try:
            FPRate = contingencyTable[1] / (contingencyTable[1] + contingencyTable[3])
        except ZeroDivisionError:
            FPRate = 0
        return FPRate
    
    def calcFMeasure(self, contingencyTable):
        try:
            fMeasure = (2 * self.calcPrecision(contingencyTable) * self.calcRecall(contingencyTable)) 
                / (self.calcPrecision(contingencyTable) + self.calcRecall(contingencyTable))
        except ZeroDivisionError:
            fMeasure = float('nan')
        return fMeasure


if __name__ == "__main__"
    testFile = 'T2_3076_num_molGeom.xyz'
    totalProcessed = 0
    totalFiles = len(os.listdir(directory))

    accuracyScores = []
    precionScores = []
    recallScores = []
    falsePositiveRates = []
    fMeasureScores = []

    timeStart = time.time()

    for filename in os.listdir(directory):
        # Loop through each file
        pointcloud = pc.PointCloud(directory + filename, inputType='xyz')
        clustered = kMeansClassifier(pointcloud)

        totalProcessed += 1

        accuracyScores.append(clustered.accuracy)
        precionScores.append(clustered.precision)
        recallScores.append(clustered.recall)
        falsePositiveRates.append(clustered.falsePositiveRate)
        fMeasureScores.append(clustered.fMeasure)

        print('Percentage complete: ' + str((totalProcessed / totalFiles) * 100))
        print('For File: ' + filename +
            ' containing ' + str(pointcloud.numMols) + ' molecules')
        print('Accuracy: ' + str(clustered.accuracy) +
            '\nPrecision: ' + str(clustered.precision) +
            '\nRecall: ' + str(clustered.recall) +
            '\nFalse Positive Rate: ' + str(clustered.falsePositiveRate) +
            '\nfMeasure: ' + str(clustered.fMeasure))
    print(str(totalFiles) + ' processed in ' + str(time.time() - timeStart) + 's')

    print('Average Accuracy: ' + str(sum(accuracyScores)/totalFiles) +
            '\nPrecision: ' + str(sum(precionScores)/totalFiles) +
            '\nRecall: ' + str(sum(recallScores)/totalFiles) +
            '\nFalse Positive Rate: ' + str(sum(falsePositiveRates)/totalFiles) +
            '\nfMeasure: ' + str(sum(fMeasureScores)/totalFiles))