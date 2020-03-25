from __future__ import print_function
import sys
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#sys.path.insert(0,'Keras-2.0.8')
from keras import backend as K
import numpy as np
from medpy.io import load,save
from keras.optimizers import SGD
from hybridnet_lc import dense_rnn_net
from denseunet3d_lc import denseunet_3d
#from denseunet import DenseUNet
from loss import weighted_crossentropy12
#from loss import weighted_crossentropy_2ddense
from lib.funcs_3d import predict_tumor_inwindow
from keras.utils2.multi_gpu import make_parallel
from scipy import ndimage
from skimage import measure
import argparse
from pathlib import Path
import pdb
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
K.set_image_dim_ordering('tf')  # Tensorflow dimension ordering in this code

#  global parameters
parser = argparse.ArgumentParser(description='Keras DenseUnet Test')
#  data folder
parser.add_argument('-data', type=str, default='/media/buyizhiyou/zly/liverdata/data/myTestData2/test-volume-', help='test images')
parser.add_argument('-liver_path', type=str, default='/media/buyizhiyou/zly/liverdata/livermask/')
parser.add_argument('-save_path', type=str, default='/media/buyizhiyou/zly/liverdata/result/test/hff_lc_12/')
#parser.add_argument('-save_path', type=str, default='/media/buyizhiyou/zly/liverdata/test_livermask/')
#  other paras
parser.add_argument('-b', type=int, default=1)
parser.add_argument('-input_size', type=int, default=256)
#parser.add_argument('-model_weight', type=str, default='/media/buyizhiyou/zly/liverdata/result/HD_lc_12/3d/H_model/weights.12-0.04.hdf5')
parser.add_argument('-model_weight', type=str, default='./model/weights-hff_lc_12_q.25-0.04.hdf5')
parser.add_argument('-input_cols', type=int, default=12)

#  data augment
parser.add_argument('-mean', type=int, default=48)
#
parser.add_argument('-thres_liver', type=float, default=0.5)
parser.add_argument('-thres_tumor', type=float, default=0.9)
args = parser.parse_args()


def predict(args):

    if not Path(args.save_path).exists():
        os.mkdir(args.save_path)

    for id in range(31, 70):
        print('-' * 30)
        print('Loading model and preprocessing test data...' + str(id))
        print('-' * 30)
        #model = denseunet_3d(args)
        model = dense_rnn_net(args)
        #model = DenseUNet(reduction=0.5, args=args)
        #import pdb
        #pdb.set_trace()
        model.load_weights(args.model_weight)
        sgd = SGD(lr=1e-2, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss=[weighted_crossentropy12])

        #  load data
        img_test, img_test_header = load(args.data + str(id) + '.nii')
        img_test -= args.mean
        #pdb.set_trace()

        #  load liver mask
        mask, mask_header = load(args.liver_path + str(id) + '-ori.nii')
        #mask, mask_header = load(args.liver_path + 'test-segmentation-' + str(id) + '.nii')
        #print(mask.shape)
        #print(mask.max())
        mask[mask==2]=1
        mask = ndimage.binary_dilation(mask, iterations=1).astype(mask.dtype)
        index = np.where(mask==1)
        #print(index)
        mini = np.min(index, axis = -1)
        #print(mini)
        maxi = np.max(index, axis = -1)
        #print(maxi)

        print('-' * 30)
        print('Predicting masks on test data...' + str(id))
        print('-' * 30)
        #import pdb
        score1, score2 =  predict_tumor_inwindow(model, img_test, 3, mini, maxi, args)
        K.clear_session()
        
        result1 = score1
        result2 = score2
        result1[result1>=args.thres_liver]=1
        result1[result1<args.thres_liver]=0
        result2[result2>=args.thres_tumor]=1
        result2[result2<args.thres_tumor]=0
        result1[result2==1]=1
       

        print('-' * 30)
        print('Postprocessing on mask ...' + str(id))
        print('-' * 30)

        #  preserve the largest liver
        Segmask = result2
        box=[]
        [liver_res, num] = measure.label(result1, return_num=True)
        region = measure.regionprops(liver_res)
        for i in range(num):
            box.append(region[i].area)
        label_num = box.index(max(box)) + 1
        #label_num = box.index(max((box), default=0)) + 1
        liver_res[liver_res != label_num] = 0
        liver_res[liver_res == label_num] = 1

        #  preserve the largest liver
        mask = ndimage.binary_dilation(mask, iterations=1).astype(mask.dtype)
        box = []
        [liver_labels, num] = measure.label(mask, return_num=True)
        region = measure.regionprops(liver_labels)
        for i in range(num):
            box.append(region[i].area)
        label_num = box.index(max(box)) + 1
        #label_num = box.index(max((box), default=0)) + 1
        liver_labels[liver_labels != label_num] = 0
        liver_labels[liver_labels == label_num] = 1
        liver_labels = ndimage.binary_fill_holes(liver_labels).astype(int)


        #  preserve tumor within ' largest liver' only
        Segmask = Segmask * liver_labels
        Segmask = ndimage.binary_fill_holes(Segmask).astype(int)
        Segmask = np.array(Segmask,dtype='uint8')
        liver_res = np.array(liver_res, dtype='uint8')
        liver_res = ndimage.binary_fill_holes(liver_res).astype(int)
        liver_res[Segmask == 1] = 2
        liver_res = np.array(liver_res, dtype='uint8')
        save(liver_res, args.save_path + 'test-segmentation-' + str(id) + '.nii', img_test_header)

        del  Segmask, liver_labels, mask, region,label_num,liver_res


if __name__ == '__main__':
    predict(args)

