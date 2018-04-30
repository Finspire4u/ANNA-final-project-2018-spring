from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
#from scipy.ndimage import rotate
#import scipy.ndimage
#from scipy import ndimage
#from random import randint
import os
#import SimpleITK as sitk
from V_net_utils import get_v_net, dice_loss, dice
import tensorflow as tf

def check_make_file(file):
	if not os.path.exists(file):
		os.makedirs(file)

print("Build network")

img_x = 128
img_y = 128
img_z = 16
bs = 5
x = tf.placeholder(tf.float32,[bs,img_z,img_x,img_y,1])
y = tf.placeholder(tf.float32,[bs,img_z,img_x,img_y,1])
y_pred = get_v_net(x)
loss = dice_loss(y_pred,y)
dc_mean = dice(y_pred,y)
optimizer = tf.train.AdamOptimizer(0.0002)
train_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

training_img = np.load('./crop_data.npz')['img_crop']
training_seg = np.load('./crop_data.npz')['seg_crop']
training_img = training_img[:,:,:,:,np.newaxis]
training_seg = training_seg[:,:,:,:,np.newaxis] #59*24 128 128 1

sess = tf.Session()
sess.run(init)
step = 0
echo = 200
print ("begin to train")
saver = tf.train.Saver(max_to_keep=500)

while step <= echo:
	for kk in range(training_img.shape[0]//bs):
		a,b = sess.run([loss,train_op], feed_dict={x: training_img[kk*bs:bs*(kk+1),:,:,:,:],
						y: training_seg[kk*bs:bs*(kk+1),:,:,:,:]})
		print('step {} loss {}'.format(step,a))
	step = step+1
