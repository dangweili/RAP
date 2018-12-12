from __future__ import print_function
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import caffe

def caffenet(source, loss_param, batch_size, include, shuffle, mirror):
    n = caffe.NetSpec()
    n.data, n.label = L.MultiImageData(include=include,
                               transform_param=dict(mirror=mirror, crop_size=227, mean_value=[104, 117, 123]),
                               multiimage_data_param=dict(root_folder="./../../data/RAP_dataset/", shuffle=shuffle, batch_size=batch_size, source=source, new_height=256, new_width=256),
                               ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=11, num_output=96, stride=4, pad=0, group=1,
                            param=[dict(name='conv1_w', lr_mult=1, decay_mult=1),
                                   dict(name='conv1_b', lr_mult=2, decay_mult=0)])
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.pool1 = L.Pooling(n.relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)

    n.conv2 = L.Convolution(n.norm1, kernel_size=5, num_output=256, stride=1, pad=2, group=2,
                            param=[dict(name='conv2_w', lr_mult=1, decay_mult=1),
                                   dict(name='conv2_b', lr_mult=2, decay_mult=0)])
    n.relu2 = L.ReLU(n.conv2, in_place=True)
    n.pool2 = L.Pooling(n.relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)

    n.conv3 = L.Convolution(n.norm2, kernel_size=3, num_output=384, stride=1, pad=1, group=1,
                            param=[dict(name='conv3_w', lr_mult=1, decay_mult=1),
                                   dict(name='conv3_b', lr_mult=2, decay_mult=0)])
    n.relu3 = L.ReLU(n.conv3, in_place=True)

    n.conv4 = L.Convolution(n.relu3, kernel_size=3, num_output=384, stride=1, pad=1, group=2,
                            param=[dict(name='conv4_w', lr_mult=1, decay_mult=1),
                                   dict(name='conv4_b', lr_mult=2, decay_mult=0)])
    n.relu4 = L.ReLU(n.conv4, in_place=True)

    n.conv5 = L.Convolution(n.relu4, kernel_size=3, num_output=256, stride=1, pad=1, group=2,
                            param=[dict(name='conv5_w', lr_mult=1, decay_mult=1),
                                   dict(name='conv5_b', lr_mult=2, decay_mult=0)])
    n.relu5 = L.ReLU(n.conv5, in_place=True)
    n.pool5 = L.Pooling(n.relu5, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    n.fc6 = L.InnerProduct(n.pool5, num_output=4096,
                           param=[dict(name='fc6_w', lr_mult=1, decay_mult=1),
                                  dict(name='fc6_b', lr_mult=2, decay_mult=0)],
                           weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    n.relu6 = L.ReLU(n.fc6, in_place=True)
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7 = L.InnerProduct(n.drop6, num_output=4096,
                           param=[dict(name='fc7_w', lr_mult=1, decay_mult=1),
                                  dict(name='fc7_b', lr_mult=2, decay_mult=0)],
                           weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    n.relu7 = L.ReLU(n.fc7, in_place=True)
    n.drop7 = L.Dropout(n.relu7, in_place=True)
    n.fine_fc8 = L.InnerProduct(n.drop7, num_output=1,
                                param=[dict(name='fc8_body_w', lr_mult=1, decay_mult=1),
                                       dict(name='fc8_body_b', lr_mult=2, decay_mult=0)],
                                weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    n.multilabel_loss = L.MultiLabelLoss(n.fine_fc8, n.label, loss_weight=1.0, multilabel_loss_param=dict(weight=loss_param))
    n.multilabel_accuracy = L.MultiLabelAccuracy(n.fine_fc8, n.label)

    return n.to_proto()


def caffenet_solver(partition, att_idx):
   s = caffe_pb2.SolverParameter()
   path = './prototxts/CaffeNet/'
   solver_file = path + 'rap2_solver_trainval_%d_%d.prototxt' % (partition, att_idx)
   s.train_net = path + 'rap2_trainval_%d_%d.prototxt' % (partition, att_idx)
   s.test_net.append(path + 'rap2_test_%d_%d.prototxt' % (partition, att_idx))
   s.test_interval = 2000
   s.test_iter.append(170)
   s.display = 20
   s.average_loss = 20
   s.base_lr = 0.0001
   s.lr_policy = "step"
   s.gamma = 0.1
   s.stepsize = 20000
   s.max_iter = 50000
   s.momentum = 0.9
   s.weight_decay = 0.0005
   s.snapshot = 50000
   s.snapshot_prefix = "temp_models/CaffeNet/rap2_trainval_part%d_%d" % (partition, att_idx)
   s.solver_mode = caffe_pb2.SolverParameter.GPU
   with open(solver_file, 'w+') as f:
        f.write(str(s)) 

def make_net():
    """
    loss_param: the weight for different databases
    lr: the learning rate for different parts
    include: training or test
    """
    # partion = 1 
    dataset = 'rap2'
    for partion in range(1, 6, 1):
        fid = open('./../static/images-list/%s_trainval_weight_%d.txt' % (dataset, partion))
        loss_params = [float(v) for v in fid.readlines()]
        fid.close()
        for idx in range(1,55,1):
            stage = 'trainval'
            source = './../static/images-list-binary/rap2_%s_%d_att%d.txt' % (stage, partion, idx)
            loss_param = loss_params[idx-1]
            batch_size = 256 
            include = dict(phase=caffe.TRAIN) 
            shuffle = True
            mirror = True
            with open('./prototxts/CaffeNet/%s_%s_%d_%d.prototxt' % (dataset, stage, partion, idx), 'w') as f:
                print(caffenet(source=source, loss_param=loss_param, batch_size=batch_size, include=include, shuffle=shuffle, mirror=mirror), file=f)
        for idx in range(1,55,1):
            stage = 'test'
            source = './../static/images-list-binary/rap2_%s_%d_att%d.txt' % (stage, partion, idx)
            loss_param = loss_params[idx-1]
            batch_size = 64
            include = dict(phase=caffe.TEST) 
            shuffle = False 
            mirror = False
            with open('./prototxts/CaffeNet/%s_%s_%d_%d.prototxt' % (dataset, stage, partion, idx), 'w') as f:
                print(caffenet(source=source, loss_param=loss_param, batch_size=batch_size, include=include, shuffle=shuffle, mirror=mirror), file=f)
        for idx in range(1, 55, 1):
            caffenet_solver(partion, idx)

if __name__ == '__main__':
    make_net()
