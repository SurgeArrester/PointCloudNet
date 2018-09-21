#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 17:28:28 2018

@author: cameron
"""
from __future__ import division

import os
import time

import numpy as np
import matplotlib.pyplot as plt

totalReflections = 0
reflectionsList = []

class SvdAligner():
    def __init__(self, alignmentCoord = 'T2_4170_num_molGeom.xyz', referenceCoord = 'T2_4170_num_molGeom.xyz'):
        '''
        Take two cif files of the T2 molecule and rotate/translate them so that they are aligned
        via the first molecules using the SVD technique. File T2_4170 as our reference as this is the only unit 
        cell with a single T2 molecule
        '''

        global totalReflections
        global reflectionsList
        
        self.fileNameOne = alignmentCoord
        self.fileNameSecond = referenceCoord
        
        self.pc1 = np.loadtxt(alignmentCoord)
        self.pc2 = np.loadtxt(referenceCoord)
        # self.pc2[:, 0] += 1 # added [1, 0, 0] for testing, makes no difference to end result
        
        # Take the first 32 atoms for rotational purposes, these represent the 
        # first molecule in each file
        self.pc1Reduced = self.pc1[:32, :] 
        self.pc2Reduced = self.pc2[:32, :]
        
        # R and T of the molecules to align the two
        self.rotationMatrix, self.translationVector = self.svdRotMatix(self.pc1Reduced, self.pc2Reduced)
        
        # If the determinant of R is 1 then it is a success and we apply the transformation 
        # Successful for 5530 molecules
        if type(self.rotationMatrix) == np.ndarray:
            self.alignedCoord = self.rotationMatrix.dot(self.pc1.T)
            self.alignedCoord -= self.translationVector.reshape(3,1)
            
        # Else  if it is -1 then they can't be aligned as they are reflections of one another
        # therefore we mirror it in the x axis. This works for all remaining T2 molecules (158 of them)
        else:
            totalReflections += 1
            reflectionsList.append(self.fileNameOne)
            mirrorAxis = np.array([[-1, 0, 0], [0, 1, 1], [0, 0, 1]])
            self.rotationMatrix, self.translationVector = self.svdRotMatix(self.pc1Reduced, self.pc2Reduced, mirrorAxis = mirrorAxis)
            self.alignedCoord = self.rotationMatrix.dot(self.pc1.T)
            self.alignedCoord -= self.translationVector.reshape(3,1)
            
        
    def svdRotMatix(self, pc1, pc2, mirrorAxis = np.identity(3)):
        ''' 
        Run through the SVD rotational method
        '''
        # Using notation from http://post.queensu.ca/~sdb2/PAPERS/PAMI-3DLS-1987.pdf


        # First we take the means of each and subtract to get to same centroid of each 
        # thus reducing translation vector to zero, which we can later calculate
                
        p = mirrorAxis.dot(np.copy(pc1.T))
        pBar = np.copy(pc2.T)
        
        q = p - p.mean(axis=1).reshape(3, 1)
        qBar = pBar - pBar.mean(axis=1).reshape(3, 1) 
        
        # We then dot product our two point clouds to get a 3x3 matrix
        S = q.dot(qBar.T)
        
        # Take the SVD of this
        u, sigma, v = np.linalg.svd(S)
        
        # Find our rotational matrix with v * ut
        calculatedRot = v.T.dot(u.T)      
        det = np.linalg.det(calculatedRot)
        
        if det > 0: # if det = 1 then it is a feasible rotation
            rotationMatrix = calculatedRot
             # we apply the rotation to align with the second cloud and subtract the two
             # to find our translation vector
            translation = pBar - rotationMatrix.dot(p)
            return rotationMatrix, translation[:,0]
            
        else:   # else not a feasible rotation and will be rerun after reflection
            return None, None
    
    def generateOutputFile(self, outputPath, coords):
        '''
        save coordinate file to a chosen location
        '''
        fileName = self.fileNameOne.split('.')[0] # remove the .xyz from the filename
        np.savetxt(outputPath + '/' + fileName + '_aligned.xyz', coords.T, delimiter = " ") # write xyz coords
        
xyzFolder = '/home/cameron/Desktop/xyz/coords'
outputFolder = '/home/cameron/Desktop/aligned'

os.chdir(xyzFolder)

success = 0
failure = 0

timeStart = time.time()
filesToProcess = len(os.listdir(xyzFolder))

for filename in os.listdir(xyzFolder):    
    alignment = SvdAligner(alignmentCoord = filename)
    
    if type(alignment.rotationMatrix) == np.ndarray:
        alignment.generateOutputFile(outputFolder, alignment.alignedCoord)
        success += 1
    
    else:
        failure += 1
        
    print(str((success / filesToProcess) * 100) + "% complete")
        
print("Processed " + str(success) + " files in " + str(time.time() - timeStart) + "s")
    
'''   
Previous code used for proof of concept for a 2D example

degToRad = np.pi / 180
rotationalAngle = 30

def rotate2d(inputMatrix, rotAngle):
    rotationalMatrix = np.array(([np.cos(rotAngle * degToRad), np.sin(rotAngle * degToRad)], [-np.sin(rotAngle * degToRad), np.cos(rotAngle * degToRad)]))
    transformedMatrix = rotationalMatrix.dot(inputMatrix)
    return transformedMatrix

def returnRotationMatrix(rotAngle):
    rotationalMatrix = np.array(([np.cos(rotAngle * degToRad), np.sin(rotAngle * degToRad)], [-np.sin(rotAngle * degToRad), np.cos(rotAngle * degToRad)]))
    return rotationalMatrix

def translate2d(inputMatrix, translationX, translationY):
    transformedMatrix = np.copy(inputMatrix)
    transformedMatrix[0, :] += translationX
    transformedMatrix[1, :] += translationY
    return transformedMatrix

def pltSct(inputMatrix):
    plt.scatter(inputMatrix[0, :], inputMatrix[1, :])
    
triangle = np.array(([0, 0], [3, 0], [2, 1])).T # To make triangle[i] a 2x1 column vector

transformedTriangle = rotate2d(triangle, rotationalAngle)
translatedTriangle = translate2d(triangle, 2, 3)


# Here R is a 2x2 rotation matrix and T is a 2x1 column vector [2, 3]

p = np.copy(triangle)
pBar = translate2d(rotate2d(p, rotationalAngle), 2, 3)


# By removing the means we simply have two matrices that have a rotational difference

q = p - p.mean(axis=1).reshape(2, 1)
qBar = pBar - pBar.mean(axis=1).reshape(2, 1) 

qTest = rotate2d(q, 90) # this should be the same as qBar

S = q.dot(qBar.T)

u, sigma, v = np.linalg.svd(S)

calculatedRot = v.T.dot(u.T)

print(returnRotationMatrix(rotationalAngle))
print(calculatedRot)
'''
