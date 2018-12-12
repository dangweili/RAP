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
from scipy.io import savemat, loadmat
import multiprocessing
from multiprocessing import Process

def sub_worker(db_file='', mat_file='', region=[0,1]):
    ''' load the predicted results '''
    lmdb_env = lmdb.open( db_file )
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    iter_cnt = 0
    for i in range(region[0], region[1]):
        if (i+1)%100 == 0:
            print i+1
        k = '%010d'%(i)
        v = lmdb_cursor.get(k)
        if v == None:
            print 'key:%s is empty'%(k)
            break
        datum.ParseFromString( v )
        data_ = caffe.io.datum_to_array( datum )
        data_ = data_.astype(np.float32, copy=False)
        L = data_.size
        if iter_cnt == 0:
            data = np.zeros((region[1]-region[0], L), dtype=np.float32)
        data_ = data_.reshape((1,L))
        data[iter_cnt,:] = data_
        iter_cnt += 1
    savemat(mat_file, {'images_feat': data})
def parse_lmdb_to_mat(db_file='', mat_file='', Cnt=1, ProcessNum=1):
    if db_file.strip() == '':
        print 'please input the lmdb file\n'
        return
    if mat_file.strip() == '':
        print 'please input the mat file\n'
        return
    pool = multiprocessing.Pool(processes = ProcessNum)
    r = int(math.ceil(Cnt*1.0/ProcessNum))
    for i in range(ProcessNum):
        region = [r*i, min(r*(i+1), Cnt)]
        mat_file_tmp = 'tmp/%d.mat'%(i+1)
        pool.apply_async(sub_worker, (db_file, mat_file_tmp, region, ))
    pool.close()
    pool.join()
    print 'parse all the lmdb to mat ok' 
    start = 0
    # pdb.set_trace()
    for i in range(ProcessNum):
        mat_file_tmp = 'tmp/%d.mat'%(i+1)
        tmp = loadmat(mat_file_tmp)
        # os.remove(mat_file_tmp)
        data_ = tmp['images_feat']
        if i==0:
            data = np.zeros((Cnt, data_.shape[1]), dtype=np.float32)
            data[start:start+data_.shape[0],:] = data_
        else:
            data[start:start+data_.shape[0],:] = data_
        start = start + data_.shape[0]
    savemat(mat_file, {'images_feat': data})

    print 'all the mat file has been ensembled'

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'please input: a) db file, b) Cnt to parse, c) save.mat'
        sys.exit()
    db_file = sys.argv[1]
    Cnt = int(sys.argv[2])
    mat_file = sys.argv[3]

    data = parse_lmdb_to_mat(db_file, mat_file, Cnt, 12)
