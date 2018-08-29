import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import time
import csv

# import diffpy.Structure
import CifFile
from scipy.spatial import distance
import networkx as nx


class PointCloud(object):
    def __init__(self, filePath,
                 extendedCloud=False,
                 netInput=False,
                 inputType='cif'):
        self.filePath = filePath
        self.fileName = filePath.split('/')[-1]  # tail of filepath for name
        self.inputType = inputType

        if inputType == 'cif':
            self._loadCifFile(filePath, extendedCloud, netInput)

        elif inputType == 'xyz':
            self.cartesianCoords = np.loadtxt(filePath)
            self.reducedCloudCart = self._generateReducedCloud(self.cartesianCoords, inputType=inputType)
        # as there are 46 atoms in a T2 molecule
        self.numMols = int(self.cartesianCoords.shape[0] / 46)

        # UNUSED Generate a voxel input for the neural net with default values
        if netInput:
            self.generateVoxelNetInput(self.cartesionCoords)

    def _loadCifFile(self, cifFilePath, extendedCloud=False,
                     netInput=False, inputType='cif'):
        '''
        Load a cif file, and generate the basic properties that we will be
        investigating
        '''
        self.filePath = cifFilePath
        self.cifFile = CifFile.ReadCif(self.filePath)  # Read cif in as a dictionary
        self.lattice = self._generateLattice(self.cifFile)  # self.molecule.lattice   # cell lengths and angles
        self.fractionalCoords = self._generateCoords(self.cifFile)  # diffpy is wrong siomehow np.array(self.molecule.xyz)
        self.transformationMatrix = self._transformationMatrix(self.lattice)
        self.cartesianCoords = self.fractionalCoords.dot(self.transformationMatrix)
        self.reducedCloudFrac, self.reducedCloudCart = self._generateReducedCloud(self.fractionalCoords, inputType=inputType)
        if extendedCloud:           # Generate an extended cell, off by default, needs lattice
            self.generateExtendedCloud(self.lattice)
        # self.molecule = diffpy.Structure.loadStructure(self.filePath)

    def _generateLattice(self, cf):
        '''
        Pull lattice data from cif file and return as a dictionary
        '''
        keys = list(cf.keys())
        keys.remove('global')  # There are two keys, global and data, we just want data
        lattice = {}
        lattice['a'] = float(cf[keys[0]]['_cell_length_a'])
        lattice['b'] = float(cf[keys[0]]['_cell_length_b'])
        lattice['c'] = float(cf[keys[0]]['_cell_length_c'])
        lattice['alpha'] = float(cf[keys[0]]['_cell_angle_alpha'])
        lattice['beta'] = float(cf[keys[0]]['_cell_angle_beta'])
        lattice['gamma'] = float(cf[keys[0]]['_cell_angle_gamma'])
        lattice['volume'] = float(cf[keys[0]]['_cell_volume'])
        return lattice

    def _generateCoords(self, cf):
        '''
        Stack the xyz fractional coords and return as a numpy aray
        '''
        keys = list(cf.keys())
        keys.remove('global')  # Just take the data block
        xs = np.array(cf[keys[0]]['_atom_site_fract_x'], np.float32)
        ys = np.array(cf[keys[0]]['_atom_site_fract_y'], np.float32)
        zs = np.array(cf[keys[0]]['_atom_site_fract_z'], np.float32)
        fractionalCoords = np.stack((xs, ys, zs), axis=1)
        return fractionalCoords

    def _transformationMatrix(self, lattice):
        '''
        This is a constant that is used to convert from fractional to 
        cartesian coordinates
        '''
        transformationMatrix = np.zeros((3, 3))
        transformationMatrix[0][0] = lattice['a']
        transformationMatrix[0][1] = lattice['b'] * np.cos(np.pi * lattice['gamma'] / 180)
        transformationMatrix[0][2] = lattice['c'] * np.cos(np.pi * lattice['beta'] / 180)
        transformationMatrix[1][1] = lattice['b'] * np.sin(np.pi * lattice['gamma'] / 180)
        transformationMatrix[1][2] = lattice['c'] * (np.cos(np.pi*lattice['alpha'] / 180) - np.cos(np.pi * lattice['beta'] / 180) * np.cos(np.pi * lattice['gamma'] / 180)) / np.sin(np.pi * lattice['gamma'] / 180)
        transformationMatrix[2][2] = lattice['volume'] / (lattice['a'] * lattice['b'] * np.sin(np.pi * lattice['gamma'] / 180))
        return transformationMatrix

    def _generateReducedCloud(self, inputCoords, inputType='cif'):
        '''
        There are 46 atoms in a single T2 molecule we can use this to find the centre of each molecule
        as an average of the 10th and 23rd atoms. As the 16 H atoms are at the end of the file, we 
        simply iterate over multiples of 32

        Gives the reduction of a single unit cell, must be extended
        '''
        self.numMols = int(inputCoords.shape[0] / 46)  # cast to int for windows compatibility
        # print('Num of molecules: ' + str(self.numMols))

        if inputType == 'cif':
            reducedCloudFrac = np.zeros((self.numMols, 3))  # create empty matrix for coords
            for i in range(self.numMols):
                reducedCloudFrac[i] = (inputCoords[(32 * i) + 9] + inputCoords[(32 * i) + 22]) * 0.5
            return reducedCloudFrac, reducedCloudFrac.dot(self.transformationMatrix)

        elif inputType == 'xyz':
            reducedCloudCart = np.zeros((self.numMols, 3)) # create empty matrix for coords
            for i in range(self.numMols):
                reducedCloudCart[i] = (inputCoords[(32 * i) + 9] + inputCoords[(32 * i) + 22]) * 0.5
            return reducedCloudCart
            
        
    def generateExtendedCloud(self, lattice, coords = None, reducedCloud = True, inputType = 'cif'):
        '''
        Extend the cloud in 7 directions, due to the mirror symmetry this is sufficient as we will be removing duplicate values
        
        Once extended calculate the adjacency matrix for the entire cloud and remove unnecessary repeated values
        '''
        
        if coords is None and reducedCloud is True:
            coords = self.reducedCloudFrac
        else:
            coords = self.cartesianCoords
            
        fractionalAxis = np.array([[lattice['a'], 0, 0], [0, lattice['b'], 0], [0, 0, lattice['c']]])
        cartesianAxis = fractionalAxis.dot(self.transformationMatrix) #  create unit cell 
        
        extendedCloud = np.concatenate((coords,     # Expand cloud across x, y and z axis
                                       coords + cartesianAxis[0],  # x
                                       coords + cartesianAxis[1],  # y
                                       coords + cartesianAxis[2],  # z
                                       coords + cartesianAxis[0] + cartesianAxis[1],  # x + y
                                       coords + cartesianAxis[0] + cartesianAxis[2],  # x + z
                                       coords + cartesianAxis[1] + cartesianAxis[2],  # y + z
                                       coords + cartesianAxis[0] + cartesianAxis[1] + cartesianAxis[2]), # x + y + z
                                       axis = 0)
        self.extendedCloud = extendedCloud # save as a property

        adjacencyMatrix = self.generateAdjacencyMatrixNN(coords = extendedCloud, knn = extendedCloud.shape[0]) # Full adjacency matrix of cloud
        adjacencyMatrix[self.numMols:, :] = 0 # replace all distances outside the first unit cell with zero
        simplifiedAdjacencyMatrix = np.triu(adjacencyMatrix[:self.numMols, :]) # remove the inner cell repeated adjacencies (upper triangle only)
        flattenedDistances = np.sort(simplifiedAdjacencyMatrix.flatten()) # vector of distances from initial cell
        self.simplifiedMinimumDist = flattenedDistances[np.nonzero(flattenedDistances)] # take non zero elements and save
        # print(self.simplifiedMinimumDist)
        
    def generateTarget(self, coords):
        target = np.zeros(coords.shape[0], dtype='int')
        for i, point in enumerate(coords):
            target[i] = i - ((i // pc.numMols) * pc.numMols)
        self.target = target
               
    def generateAdjacencyMatrixNN(self, coords = None, knn = 10):  
        '''
        create an adjacency matrix of the k nearest neihbours for each atom in the cell
        '''
        if coords is None:
            coords = self.cartesianCoords
            
        self.distanceMatrix = distance.pdist(coords, 'euclidean') # Create condensed distance matrix
        self.adjacencyMatrix = distance.squareform(self.distanceMatrix) # Create standard adjacency matrix
        
        for i, row in enumerate(self.adjacencyMatrix):
            lowestVals = np.partition(row, knn-1)[:knn] # take k neighbours from that row
            threshold = lowestVals.max()    # Take the longest distance from k neighbours
            exceedsThresholdFlags = row > threshold # take indices of all values greater than threshold
            self.adjacencyMatrix[i][exceedsThresholdFlags] = 0 # take them to zero
        return self.adjacencyMatrix
            
    def generateOutputFile(self, outputFolder, coords = None, generateLattice = True, fileExtension = 'xyz', newLineChar="\n"):
        '''
        Given a path, write the given matrix and cartesian lattice (for future 
        expansion if necessary)
        
        '''
        if type(coords) == None:
            coords = self.cartesianCoords
            
        fileName = self.fileName.split('.')[0]
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        np.savetxt(outputFolder + fileName + fileExtension, coords, delimiter = " ", newline=newLineChar) # write matrix/vector
        
        # Lattice File Generation
        if generateLattice:
            if not os.path.exists(outputFolder + 'lattice/'):
                os.makedirs(outputFolder + 'lattice/')
                
            with open(outputFolder + 'lattice/' + fileName + '.lat', "wb") as latticeFile:
                w = csv.writer(latticeFile, delimiter = " ") 
                for key, val in self.lattice.items():
                    w.writerow([key, val])
                w.writerow(["Cartesian A", np.array([self.lattice['a'], 0, 0]).dot(self.transformationMatrix)])
                w.writerow(["Cartesian B", np.array([0, self.lattice['b'], 0]).dot(self.transformationMatrix)])
                w.writerow(["Cartesian C", np.array([0, 0, self.lattice['c']]).dot(self.transformationMatrix)])
            
    def plot(self, coords):
        '''
        Plot a pointcloud based on given coordinates
        '''
        xs = coords[:,0]
        ys = coords[:,1]
        zs = coords[:,2]
        
        fig = plt.figure()
        ax = Axes3D(fig)
        
        ax.scatter(xs, ys, zs)

        ax.view_init(elev=50, azim=230)
        plt.show()
        
    ##########################################################################
    ####        Below functions unused in final implementations           ####
    ##########################################################################

    def generateVoxelNetInput(self, coords, divisions = 1000, writeToFile = False, outputFilePath = '/netInputs/'):
        '''
        Generate a voxel grid of atoms, infeasible due to massive size
        '''
        # netInput = np.zeros(divisions, divisions, divisions, dtype = int) # empty array for the net input
        normalizedCoords = np.zeros(coords.shape, dtype = int) # empty array for normalised coords
        xmax, xmin = coords.max(), coords.min()
        normalizedCoords = np.array(((coords - xmin) / (xmax - xmin) * divisions), dtype = int) # normalize coords
        indexingArray = np.zeros(normalizedCoords.shape[0], dtype=int)
        for i, row in enumerate(normalizedCoords):
            indexingArray[i] = row[0] + (row[1] * divisions) + (row[2] * divisions * divisions)
        
        self.sortedIndexArray = np.sort(indexingArray) # which input neurons we should fire
        
        if writeToFile:
            f = open(self.fileName + 'net', 'wb')
            currentIndex = 0
            for i in range(divisions):
                for j in range(divisions):
                    for k in range(divisions):
                        if currentIndex in self.sortedIndexArray:
                            f.write('1 ')   # write a one for present atoms
                        else:
                            f.write('0 ')   # else write a zero
                print(i)
            f.close()
       
    def atomicFingerPrint(self, adjacencyMatrixRow, k=12):
        '''
        This was the atomic fingerprint function as used in paper on variational autoencoders
        This was unused in the end
        '''
        # Create adjacency matrix for entire cloud
        if self.adjacencyMatrix is None:
            # Take the first point in the file
            neighbourDistances = self.createAdjacencyMatrix(knn = self.cartesianCoords.shape[0])[0]
        else:
            # Take in the given row
            neighbourDistances = adjacencyMatrixRow
            
        # Remove our zero value for the atom in question
        indexOfRoot = np.argmin(neighbourDistances) # find index (TODO Remove?)
        neighbourDistances = np.delete(neighbourDistances, indexOfRoot)  # remove
        
        # Take the minimum distance Ri0
        Rmin = np.amin(neighbourDistances)
        Rmax = Rmin * 6
        
        # Remove all distances greater than Rmax
        neighbourDistances = neighbourDistances[neighbourDistances < Rmax] 
        
        R = np.linspace(0, Rmax, num = 1024) # input to afp function
        output = np.zeros(1024)
        
        for i, radius in enumerate(R):
            summedAfp = 0
            for j in neighbourDistances:
                summedAfp += (radius - (j / Rmin))
                # TODO add kaiser bessel smeared delta function to this
            # print summedAfp
            output[i] = summedAfp
        self.Afp = output
        
    def createMST(self):
        '''
        Using the networkX library create a MST from the adjacency Matrix, this 
        was unused after initial investigation
        '''
        graph = nx.to_networkx_graph(self.adjacencyMatrix) # initialise graph
        self.mst = nx.minimum_spanning_tree(graph)

    def loadXyzFile(self, filePath):
        '''
        Open a .xyz file from a file path that has been generated by the program Mercury, 
        mostly unused after initial fortnight
        '''
        with open (filePath, "r") as myfile: 
            data=myfile.readlines()
        
        data = data[2:] # Remove first two lines
        
        # split into terms and remove first term (the element name)
        for i, line in enumerate(data):
            data[i] = line.split()[1:]
        
        xs = []
        ys = []
        zs = []
        
        for line in data:
            xs.append(float(line[0]))
            ys.append(float(line[1]))
            zs.append(float(line[2]))

if __name__ == "__main__":
    directory = '/home/cameron/Desktop/molGeom/'
    winDirectory = '../TestFiles/molGeom/'
    testFile = 'T2_1_num_molGeom.cif'
    filePath = '/home/cameron/Dropbox/T2_Dataset/molGeom/T2_1_num_molGeom.cif'
    outputFile = 'output.txt'

    coordinatePath = '/home/cameron/Desktop/xyz/'
    extendedPath = '/home/cameron/Desktop/extended/'

    k = 7
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    time_start = time.time()
    totalProcessed = 0.0
    totalFiles = len(os.listdir(directory))

    for filename in os.listdir(directory):
        print(directory + filename)
        pc = PointCloud(directory + filename)
        
        pc.generateExtendedCloud(pc.lattice, reducedCloud=True)
        pc.generateTarget(pc.extendedCloud)
        
        if not os.path.exists(extendedPath + str(pc.numMols) + '/points/'):
                    os.makedirs(extendedPath + str(pc.numMols) + '/points/')
                    
        if not os.path.exists(extendedPath + str(pc.numMols) + '/points_label/'):
                    os.makedirs(extendedPath + str(pc.numMols) + '/points_label/')
                    
        pc.generateOutputFile(extendedPath + str(pc.numMols) + '/points/', coords=pc.extendedCloud, generateLattice=False, fileExtension = '.xyz')
        pc.generateOutputFile(extendedPath + str(pc.numMols) + '/points_label/', coords=pc.target, generateLattice=False, fileExtension = '.tgt')
        totalProcessed += 1
        # pc.plot(pc.extendedCloud)
        print('Percentage complete: ' + str((totalProcessed / totalFiles) * 100))
        print('Time taken: ' + str(time.time() - time_start) + 's\n\n')
        print

    #pc = PointCloud(directory + testFile)
    #print('Time taken: ' + str(time.time() - time_start))   


    #    with open(outputFile, 'a') as output:
    #        output.write(str(filename) + " ")
    #        for i in range(k):
    #            output.write(str(np.around(pc.simplifiedMinimumDist[i], 7)) + " ")
    #        output.write("\n")
    #    pc.generateNetInput(pc.cartesianCoords, writeToFile =False)
    #        
    #    #pc.atomicFingerPrint(x[0])
    #    
    #    #fingerPrint = pc.atomicFingerPrint
    #    pc.plot(pc.cartesianCoords)
    #    pc.plot(pc.extendedCloud)
        

