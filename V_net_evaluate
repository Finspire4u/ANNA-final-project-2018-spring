from __future__ import print_function
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
#from scipy.ndimage import rotate
#import scipy.ndimage
#from scipy import ndimage
#from random import randint
#import os
#import SimpleITK as sitk
from V_net_utils import get_v_net
import tensorflow as tf

img_x = 128
img_y = 128
img_z = 16
bs = 1
x = tf.placeholder(tf.float32,[bs,img_z,img_x,img_y,1])
y = tf.placeholder(tf.float32,[bs,img_z,img_x,img_y,1])
y_pred = get_v_net(x)

#save_path = 'D:\Inspiration\OU\2-ANNA\project'
testing_img = np.load('./crop_test.npz')['img_crop']
testing_seg = np.load('./crop_test.npz')['seg_crop']
testing_img = testing_img[:,:,:,:,np.newaxis]
testing_seg = testing_seg[:,:,:,:,np.newaxis]
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
saver_vnet = tf.train.Saver(max_to_keep=500)
saver_vnet.restore(sess,'D:/Inspiration/OU/2-ANNA/project/model/model-200')
seg_result = np.zeros((10,16,128,128,1))
for kk in range(testing_img.shape[0]//bs):
	seg_result[kk] = sess.run(y_pred,feed_dict={x: testing_img[kk*bs:bs*(kk+1),:,:,:,:],
							  y: testing_seg[kk*bs:bs*(kk+1),:,:,:,:]})
np.save('V_net_evaluate.npy',seg_result = seg_result)
