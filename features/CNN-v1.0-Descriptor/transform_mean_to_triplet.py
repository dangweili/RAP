#!/bin/env python

import sys
import caffe
import numpy as np
import pdb
from scipy.io import savemat
pdb.set_trace()

#mean_file = './models/CaffeNet/imagenet_mean.binaryproto'
mean_file = './models/ResNet/ResNet_mean.binaryproto'
blob_single = caffe.proto.caffe_pb2.BlobProto()
data = open(mean_file, 'rb').read()
blob_single.ParseFromString( data )
imagenet_mean = np.array( caffe.io.blobproto_to_array(blob_single) )
imagenet_mean = np.squeeze(imagenet_mean.transpose((0,3,2,1)))
#savemat('./models/CaffeNet/imagenet_mean.mat', {'imagenet_mean': imagenet_mean })
savemat('./models/ResNet/ResNet_mean.mat', {'imagenet_mean': imagenet_mean })

