# PointCloudNet
A set of Pre-Processing and Analysis scripts used for the dissertation project of MSc

The below commands were executed on a Ubuntu based 64-bit linux system (Linux Mint 18.2) with a GTX 970 graphics card. All files in the PointNet folder should be executed on a cuda enabled system.

Create a new conda environment from the environment.yml file with the command:

conda env create --file=environment.yml --name=pointcloud

From here you can run any of the scripts in the normal fashion, Pointcloud.py will require you to update the variable "directory" to point to the folder of cif files you wish to process and the path you wish to write the processed files to in the pc.generateOutputFile() functions within the main loop

For the pointnet classification file this is run via:

python train_classification.py --workers=1 --nepoch=25 --scoresFolder="output/T2DatasetClassification" --batchSize=22 --num_points=2500 --dataset="T2Dataset"

With the scoresFolder and dataset commandline arguments to be updated for the desired dataset/outputFolder

train_segmentation.py appears to have been improved to the point of not working in this release, this will be remedied in future versions
