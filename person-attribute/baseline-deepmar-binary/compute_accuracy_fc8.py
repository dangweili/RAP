import lmdb
import caffe
from caffe.proto import caffe_pb2
import numpy as np
import scipy as sp
import os
import pdb
import math
import sys
import cPickle as pickle


def calculate_accuracy(gt_result, pt_result):
    ''' obtain the label-based and instance-based accuracy '''
    # compute the label-based accuracy
    if gt_result.shape != pt_result.shape:
        print 'Shape beteen groundtruth and predicted results are different'
    # compute the label-based accuracy
    result = {}
    gt_pos = np.sum((gt_result == 1).astype(float), axis=0)
    gt_neg = np.sum((gt_result == -1).astype(float), axis=0)
    pt_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=0)
    pt_neg = np.sum((gt_result == -1).astype(float) * (pt_result == -1).astype(float), axis=0)
    label_pos_acc = 1.0*pt_pos/gt_pos
    label_neg_acc = 1.0*pt_neg/gt_neg
    label_acc = (label_pos_acc + label_neg_acc)/2
    result['label_pos_acc'] = label_pos_acc
    result['label_neg_acc'] = label_neg_acc
    result['label_acc'] = label_acc
    # compute the instance-based accuracy
    # precision
    gt_pos = np.sum((gt_result == 1).astype(float), axis=1)
    pt_pos = np.sum((pt_result == 1).astype(float), axis=1)
    floatersect_pos = np.sum((gt_result == 1).astype(float)*(pt_result == 1).astype(float), axis=1)
    union_pos = np.sum(((gt_result == 1)+(pt_result == 1)).astype(float),axis=1)
    # avoid empty label in predicted results
    cnt_eff = float(gt_result.shape[0])
    for iter, key in enumerate(gt_pos):
        if key == 0:
            union_pos[iter] = 1
            pt_pos[iter] = 1
            gt_pos[iter] = 1
            cnt_eff = cnt_eff - 1
            continue
        if pt_pos[iter] == 0:
            pt_pos[iter] = 1
    instance_acc = np.sum(floatersect_pos/union_pos)/cnt_eff
    instance_precision = np.sum(floatersect_pos/pt_pos)/cnt_eff
    instance_recall = np.sum(floatersect_pos/gt_pos)/cnt_eff
    floatance_F1 = 2*instance_precision*instance_recall/(instance_precision+instance_recall)
    result['instance_acc'] = instance_acc
    result['instance_precision'] = instance_precision
    result['instance_recall'] = instance_recall
    result['instance_F1'] = floatance_F1
    return result
    
def load_results(db_file='', Cnt=1):
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
    net_type = sys.argv[1]
    stage = sys.argv[2]
    partion = sys.argv[3]
    test_file = './../static/images-list/rap2_%s_%s.txt'%(stage, partion)
    fid = open(test_file)
    for idx,line in enumerate(fid.readlines()):
        line = line.strip()
        if line == "":
            break
        line = line.split()[1:]
        data_ = map(float, line) # the first iterm is name, the last 35 iterms is label
        if idx == 0:
            gt_result = np.array(data_).reshape((1,len(data_)))
            continue
        data_ = np.array(data_).reshape((1,len(data_)))
        gt_result = np.concatenate((gt_result, data_), axis=0)
    # load the predicted results from 54 attributes
    Cnt = gt_result.shape[0] # number of test images
    pt_result = np.zeros(gt_result.shape)
    for idx in range(pt_result.shape[1]):
        att_idx = idx + 1
        db_file = 'lmdb/result_%s_fc8_%s_%d_lmdb'%(net_type, partion, att_idx)
        pt_result[:, [idx]] = load_results(db_file, Cnt)
    pt_result[pt_result<0] = -1 # transform the float to float
    pt_result[pt_result>=0] = 1
    # compute instance accuracy and label-based accuracy
    result = calculate_accuracy(gt_result, pt_result)
    fid = open('results/%s/result_%s_%s.pkl'%(net_type, stage, partion), 'wb')
    pickle.dump(result, fid)
    fid.close()
    # output the result floato txt file
    fid = open('./../static/rap_annotation_attribute-english.txt')
    AttributeNames = [line.strip() for line in fid.readlines()]
    fid.close()
    fid = open('./../static/selected_attribute_idx.txt')
    idx_selected = map(lambda x: int(x)-1, [line.strip() for line in fid.readlines()])
    fid.close()
    fid = open('results/%s/result_%s_%s.txt'%(net_type, stage, partion), 'w+')
    fid.write('label-based results of 54 attributes: acc_pos; acc_neg; acc_mean; attribute name;\n')
    for i in range(len(result['label_acc'])):
        fid.write('%f %f %f %s\n'%(result['label_pos_acc'][i], result['label_neg_acc'][i], result['label_acc'][i], AttributeNames[idx_selected[i]]))
    fid.write('%f %f %f %s\n'%(result['label_pos_acc'].mean(), result['label_neg_acc'].mean(), result['label_acc'].mean(), 'Average results'))
    fid.write('\n')
    fid.write('instance-based results:\n')
    fid.write('%f %s\n'%(result['instance_acc'],'instance_accuracy'))
    fid.write('%f %s\n'%(result['instance_precision'],'instance_precision'))
    fid.write('%f %s\n'%(result['instance_recall'],'instance_recall'))
    fid.write('%f %s\n'%(result['instance_F1'],'instance_F1'))
    fid.close()
    print 'all the results has been saved in results/%s/result_%s_%s.txt!'%(net_type, stage, partion)

