import numpy as np
import scipy as sp
import cPickle as pickle
import sys
import pdb


if __name__ == '__main__':
    Net = sys.argv[1]
    stage = sys.argv[2]
    results = []
    for i in range(5):
        fid = open('./results/%s/result_%s_%d.pkl'%(Net, stage, i+1), 'rb')
        data = pickle.load(fid)
        results.append(data)
    # average all the results
    for i in range(5):
        if i == 0:
            # result = results[i]
            result = dict()
            result['label_pos_acc'] = results[i]['label_pos_acc']
            result['label_neg_acc'] = results[i]['label_neg_acc']
            result['label_acc'] = results[i]['label_acc']
            result['instance_acc'] = results[i]['instance_acc']
            result['instance_precision'] = results[i]['instance_precision']
            result['instance_recall'] = results[i]['instance_recall']
            result['instance_F1'] = results[i]['instance_F1']
        else:
            # for key in result:
            #    result[key] += results[i][key]
            result['label_pos_acc'] += results[i]['label_pos_acc']
            result['label_neg_acc'] += results[i]['label_neg_acc']
            result['label_acc'] += results[i]['label_acc']
            result['instance_acc'] += results[i]['instance_acc']
            result['instance_precision'] += results[i]['instance_precision']
            result['instance_recall'] += results[i]['instance_recall']
            result['instance_F1'] += results[i]['instance_F1']
    for key in result:
        result[key] = result[key]/5.0;
    fid = open('./results/%s/result_%s_average.txt'%(Net, stage), 'w+')
    for i in range(len(result['label_acc'])):
        fid.write('%f %f %f\n'%(result['label_pos_acc'][i], result['label_neg_acc'][i], result['label_acc'][i]))
    fid.write('%f %f %f\n'%(np.mean(result['label_pos_acc']), np.mean(result['label_neg_acc']), np.mean(result['label_acc'])))
    fid.write('%f %f %f %f'%(result['instance_acc'], result['instance_precision'], result['instance_recall'], result['instance_F1']))
    fid.close()
    print np.mean(result['label_pos_acc'])
    print np.mean(result['label_neg_acc'])
    print np.mean(result['label_acc'])
    print result['instance_acc']
    print result['instance_precision']
    print result['instance_recall']
    print result['instance_F1']
