from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import progressbar
import sys
import torchvision.transforms as transforms
import argparse
import json


class PartDataset(data.Dataset): # subclass of torch datase
    def __init__(self, root, npoints = 2500, classification = False, class_choice = None, train = True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'moleculeLabel.txt')
        self.cat = {}

        self.classification = classification

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1] # make key of cat = numeric value
        print("31 " + str(self.cat))

        if not class_choice is  None: # skipped normally
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        print("37 " + str(self.cat))
        
        for item in self.cat: # loop through category keys
            print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            #print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point)) # sort filenames
            if train:                           # if training
                fns = fns[:int(len(fns) * 0.9)] # take first 90%
            else:                               # else
                fns = fns[int(len(fns) * 0.9):] # take last 10%

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.xyz'), os.path.join(dir_seg, token + '.tgt')))

        self.datapath = []
        for item in self.cat: # for each category
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1])) # append (category, inputFile.xyz, targetFile.tgt)

        
        print('line 61 Categories: ' + str(self.cat))
       # print('datapath: ' + str(self.datapath))
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        self.num_seg_classes = 0

        if not self.classification:     # runs for segmentation only
            for i in range(len(self.datapath)//50):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        #print(self.num_seg_classes)


    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]] # get class from folder 
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        # print("Seg is " + str(seg) + " and dimensions are " + str(np.ndim(seg)))
        # print(fn[2])
        # print(point_set.shape, seg.shape)

        if np.ndim(seg) == 0:  # this is weird behaviour for scalars due to internal numpy code
            choice = 0 # np.random.choice(1, self.npoints, replace=True) 
            point_set = np.array([point_set[choice]])
            seg = np.array([seg])

        else:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
            #resample
            # print(choice)
            point_set = point_set[choice, :]
            seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'])
    print(len(d))
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(),seg.type())

    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    print(len(d))
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls.size(),cls.type())
