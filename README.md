# PointCloudNet
A set of Pre-Processing and Analysis scripts used for the dissertation project of MSc using neural networks to cluster and categorise T2 molecules

To use this software, you must have git, python 2.7 and the conda package manager installed on your system. It is assumed that this will be run under a Linux based operating system with at least 5GB of free space, however this may be modified to work under windows. It is recommended to run this using a modern nVidia GPU (although not essential) to leverage the cuda capabilities in training time. Additionally it is assumed the user has available a copy of the base T2 dataset, originally obtained from Vitaliy Kurlin.

* Fork the repository from GitHub to an empty folder on your machine using
	`git clone https://github.com/SurgeArrester/PointCloudNet.git`

* item Create the environment to run the programs under by changing to your PointCloudNet folder and running
`conda env create -f environment.yml`
Once the required packages are downloaded and installed, activate by running
`source activate pointcloudnet`

* Modify the `PointCloud.py` folder to point towards your T2 dataset folder, in line 321 under the variable `directory` and additionally udating the variable `extendedPath` for where you wish the output files to be generated.

* Run the code to create the extended reduced cloud dataset initially, change the `reducedCloud` flag to `False` in line 341 to create the extended cloud dataset, and comment out line 341 and replace `pc.extendedCloud` with `pc.reducedCloudCart` in lines 350 and 352 to create the unit cell reduced cloud.

* To generate the aligned dataset update the folder path `xyzFolder` in line 103 to be either your unit cell or extended cloud dataset folder and update the output folder `outputFolder` in line 104 for where you wish to save this. Run this code and the aligned dataset will be generated in the output directory

* To run the K-Means clustering algorithm in `kMeansClustering.py`, create a folder containing the generated Cartesian dataset into a single joined folder, then update the variable `directory` in line 16 to point to this folder. Run the code and the average performance measures will be outputted to the console across the entire dataset

* To run the neural network change directory to the PointnetImplementation Folder, from here we must run a longer command, which may need tweaking depending on the performance of your system. Update the commandline variable `--dataset` to point to the path of the dataset you wish to run this on (in our example we copied all the datasets into the PointnetImplementation folder for clarity). If you get out of memory errors this is because your GPU has run out of VRAM and the batch size will need to be reduced. If you get segmentation errors then this is due to how the internal PyTorch implementation is interacting with your operating system and you may need to reduce the number of workers to 1. Both of these increase the running time, but should not affect performance dramatically. This is run on our system via:
`python train\textunderscore classification.py --workers=1 --nepoch=25 \\ --scoresFolder="output/scoreExtendedAligned" --batchSize=15 --num\textunderscore points=2500 --dataset="T2DatasetExtendedAligned"`

* Once all iterations have been completed, the loss and accuracy have been recorded in our scores folder, and additionally we have the predicted and actual classes on the test data in this scores folder. We can plot this and find the average of the final 100 results by modifying line 55 in `calculateAndPlot.py` to point to our scores folder and running.
