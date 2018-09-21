#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:53:34 2018

@author: cameron

Rearrange xyz and tgt files to their appropriate folders depending on number of mols

As the t2 cif files are organised by the non hydrogen atoms followed by the hydrogen atoms
we will first label the non-hydrogens, then the hydrogens

"""

import os
from shutil import copyfile

xyzDirectory = '/home/cameron/Desktop/aligned/'
extendedDirectory = '/home/cameron/Desktop/extendedReduced/'
targetDirectory = '/home/cameron/Desktop/aligned_target/'
testFile = ['T2_1_num_molGeom.xyz']

networkInputFolder = '/home/cameron/Desktop/PointnetImplementation/T2DatasetExtendedReduced/'


# Generate the input and target for the extended cloud
for folder in os.listdir(networkInputFolder):
    if folder != 'moleculeLabel.txt':
        for filename in os.listdir(networkInputFolder + folder + '/points'):
            moleculeName = filename.split('.')[0]
            
            if not os.path.exists(networkInputFolder + folder + '/points/'):
                os.makedirs(networkInputFolder + folder + '/points/')
                
            if not os.path.exists(networkInputFolder + folder + '/points_label/'):
                os.makedirs(networkInputFolder + folder + '/points_label/')
            
            copyfile(networkInputFolder + folder + '/points/' + moleculeName + '.min', extendedDirectory + '/' + folder + '/' + moleculeName + '.min')
            
            targetFile = networkInputFolder + folder + '/points_label/' + moleculeName + '.tgt'
            with open(targetFile, 'w') as output:
                output.write(folder)
            

## Generate the target files from xyz files (for T2)
#for filename in os.listdir(xyzDirectory):
#    count = len(open(xyzDirectory + filename).readlines()) # count the number of lines
#    numMol = count / 46 # for T2 molecules
#    newFile = targetDirectory + filename.split('.')[0] + '.tgt' # new filepath
#    
#    with open(newFile, 'w') as output:
#        for i in range(count - (numMol * 14)):
#            output.write(str(i / 32))  # won't work in 3.7 only 2.7 as treats as ints
#            output.write('\n')
#        for i in range(count - (numMol * 14), count):
#            output.write(str((i - numMol * 32) / 14))  # won't work in 3.7 only 2.7 as treats as ints
#            output.write('\n')
#    print(numMol)
#
## Separate the target files into correct directories for the neural network
#for fileName in os.listdir(targetDirectory):
#    count = len(open(targetDirectory + fileName).readlines()) # number of lines
#    numMols = str((count // 46) + 1) # floor division and add 1 to get molecule count
#    
#    if not os.path.exists(networkInputFolder + numMols):
#        os.makedirs(networkInputFolder + numMols)
#    
#    if not os.path.exists(networkInputFolder + numMols + '/points_label/'):
#            os.makedirs(networkInputFolder + numMols + '/points_label/')
#            
#    copyfile(targetDirectory + fileName, networkInputFolder + numMols + '/points_label/' + fileName)   
#    
#    