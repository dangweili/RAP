import caffe
import collections
import copy
import numpy as np
import pdb

model_save = 'model_fusion_body_locpartem_featconcat.caffemodel'

deploy_stn_bodypartem = './prototxts/MSCAN_full/rap2_val.prototxt'
model_stn_bodypartem = './temp_models/MSCAN_full/rap2_trainval_iter_50000.caffemodel'

deploy_partem = './prototxts/MSCAN_part/rap2_val.prototxt' 
model_partem = './temp_models/MSCAN_part/rap2_trainval_iter_50000.caffemodel'

deploy_body = './prototxts/MSCAN_body/rap2_val.prototxt'
model_body = './temp_models/MSCAN_body/rap2_trainval_iter_50000.caffemodel'

net_stn_bodypartem = caffe.Net(deploy_stn_bodypartem, model_stn_bodypartem, caffe.TEST)
print 'init the stn network'
pdb.set_trace()

net_partem = caffe.Net(deploy_partem, model_partem, caffe.TEST)
print 'init the loc network'
pdb.set_trace()

net_body = caffe.Net(deploy_body, model_body, caffe.TEST)
print 'init the body network'
pdb.set_trace()

print net_stn_bodypartem.params.keys()

print net_partem.params.keys()

print net_body.params.keys()

import pdb
pdb.set_trace()
tmp = dict()
for k in net_partem.params.keys():
    if net_stn_bodypartem.params.has_key(k):
        tmp[k] = 1
        for j in range(len(net_stn_bodypartem.params[k])):
            net_stn_bodypartem.params[k][j].data[...] = net_partem.params[k][j].data
    else:
        print 'net_stn_bodypartem has not the key: ' + k
for k in net_body.params.keys():
    if tmp.has_key(k):
        print 'net_stn_bodypartem has processed the key: ' + k
        continue
    if net_stn_bodypartem.params.has_key(k):
        tmp[k] = 1
        for j in range(len(net_stn_bodypartem.params[k])):
            net_stn_bodypartem.params[k][j].data[...] = net_body.params[k][j].data
    else:
        print 'net_stn_bodypartem has not the key: ' + k

# process the last fc2_full
print net_partem.params['fc2_part'][0].data.shape
print net_body.params['fc2'][0].data.shape
print net_stn_bodypartem.params['fc2_full'][0].data.shape
import pdb
pdb.set_trace()
j = 0
data = np.concatenate((net_body.params['fc2'][j].data, net_partem.params['fc2_part'][j].data), axis=1)
net_stn_bodypartem.params['fc2_full'][j].data[...] = data
#j = 0
#data = (net_body.params['fc2_body'][j].data + net_partem.params['fc2_part'][j].data)/2
#net_stn_bodypartem.params['fc2_full'][j].data[...] = data
j = 1
data = (net_body.params['fc2'][j].data + net_partem.params['fc2_part'][j].data)/2
net_stn_bodypartem.params['fc2_full'][j].data[...] = data

# save the model
net_stn_bodypartem.save(model_save)
