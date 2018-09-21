from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
from pointnet import PointNetCls
import torch.nn.functional as F



parser = argparse.ArgumentParser()   # basic parameters
parser.add_argument('--batchSize', type=int, default=32, help='input batch size') # On gtx970 runs out of memory with 32, 24 works fine
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4) # found better to be singlethreaded == 1
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for') # reduced to 10 in practice for timing
parser.add_argument('--outf', type=str, default='cls',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--scoresFolder', type=str, default = 'scoresClassification/', help='Folder for scores')
parser.add_argument('--dataset', type=str, default='T2Dataset', help='Dataset to load')

opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m' # what does this do?

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = opt.dataset, classification = True, npoints = opt.num_points) # from datasets.py
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = opt.dataset, classification = True, train = False, npoints = opt.num_points)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

# create output folder
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# use PointNet network
classifier = PointNetCls(k = num_classes, num_points = opt.num_points)

# Load model if so chosen
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

# standard optimiser
optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
classifier.cuda() # enable cuda

num_batch = len(dataset)/opt.batchSize

# Create variables to save scores
testLoss = []
testAccuracy = []

# Create scores folder if not existing
if not os.path.exists(opt.scoresFolder):
        os.makedirs(opt.scoresFolder)

# Create a folder for the scores and 
if not os.path.exists(opt.scoresFolder + "/predictions/"):
        os.makedirs(opt.scoresFolder + "/predictions/")


for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points, target = Variable(points), Variable(target[:,0]) # Points shape is: torch.Size([22, 2500])
        # print("Shape is: " + str(points.shape))
        points = points.transpose(2,1)                       # Points shape is: torch.Size([22, 3, 2500])
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        # print("Shape is: " + str(points.shape))
        pred, _ = classifier(points)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.item(),correct.item() / float(opt.batchSize)))

        if i % 10 == 0: # every ten iterations run a test on unseen data
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points, target = Variable(points), Variable(target[:,0])
            points = points.transpose(2,1)
            points, target = points.cuda(), target.cuda()
            pred, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]

            labels = target.data
            classifications = pred_choice

            correct = pred_choice.eq(target.data).cpu().sum()
            testLoss.append(loss.item())
            testAccuracy.append(correct.item()/float(opt.batchSize))
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

            with open(opt.scoresFolder + "/predictions/labelIterationEp" + str(epoch) + "it" + str(i) + ".out", "wb") as lossFile:
                for item in labels:
                    lossFile.write("{}\n".format(item))

            with open(opt.scoresFolder + "/predictions/classIterationEp" + str(epoch) + "it" + str(i) + ".out", "wb") as accuracyFile:
                for item in classifications:
                    accuracyFile.write("{}\n".format(item))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))  
        
with open(opt.scoresFolder + "/lossClassification.out", "wb") as lossFile:
    for item in testLoss:
        lossFile.write("{}\n".format(item))

with open(opt.scoresFolder + "/accuracyClassification.out", "wb") as accuracyFile:
    for item in testAccuracy:
        accuracyFile.write("{}\n".format(item))
