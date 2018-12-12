#!/usr/bin/env python

import lmdb
import caffe
from caffe.proto import caffe_pb2
import numpy as np
import scipy as sp
from scipy.io import loadmat,savemat
import os
import pdb
import math
import sys


def load_results(db_file = '', Cnt=1):
    ''' load the predicted results '''
    lmdb_env = lmdb.open( db_file )
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    for i in range(Cnt):
        k = '%010d'%(i)
        v = lmdb_cursor.get(k)
        if v == None:
            print 'key:%s is empty'%(k)
            break
        datum.ParseFromString( v )
        data_ = caffe.io.datum_to_array( datum )
        data_ = data_.astype(np.float32, copy=False)
        L = data_.size
        if i == 0:
            data = np.zeros((Cnt, L))
        data_ = data_.reshape((1,L))
        data[i,:] = data_
    return data
if __name__ == '__main__':
    # load the ground truth labels from test list
    iter_Cnt = 5
    lmdb_path = './../baseline-deepmar/lmdb/'
    for i in range(iter_Cnt):
        lmdb_name = 'result_ResNet50_fc8_%d_lmdb'%(i+1)
        test_file = './../static/images-list/rap2_test_%d.txt'%(i+1)
        save_name = './data/deepmar_resnet50_%d.mat'%(i+1)
        imgs_name = []
        fid = open(test_file)
        for idx,line in enumerate(fid.readlines()):
            line = line.strip()
            if line == "":
                break
            line = line.split()
            imgs_name.append(line[0])
            line = line[1:]
            data_ = map(float, line)
            if idx == 0:
                gt_result = np.array(data_).reshape((1, len(data_)))
                continue
            data_ = np.array(data_).reshape((1, len(data_)))
            gt_result = np.concatenate((gt_result, data_), axis=0)
        # load the predicted results
        Cnt = gt_result.shape[0]
        pt_result = load_results(lmdb_path+lmdb_name, Cnt)
        ##############################################################
        savemat(save_name, {'pt_result':pt_result, 'gt_result':gt_result, 'imgs_name': imgs_name}) 
