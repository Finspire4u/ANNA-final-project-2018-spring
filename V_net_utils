from __future__ import print_function
import numpy as np
import os
import tensorflow as tf

def conv_3d_pool(x,name,filter_num,filter_size,strides):
	with tf.variable_scope(name):
		input_channel = x.get_shape()[4]
		W = tf.get_variable("weights",[filter_size[0],filter_size[1],filter_size[2],
							input_channel,filter_num],
							initializer=tf.random_normal_initializer(stddev=0.02))
		b = tf.get_variable("bias",[filter_num],initializer=tf.constant_initializer(0.0))
		conv = tf.nn.conv3d(x,W,strides=[1,strides[0],strides[1],strides[2],1],padding='SAME')
		return tf.nn.relu(conv+b)

def conv_3d(x,name,filter_num,filter_size):
	with tf.variable_scope(name):
		input_channel = x.get_shape()[4]
		W = tf.get_variable("weights",[filter_size[0],filter_size[1],filter_size[2],
							input_channel,filter_num],
							initializer=tf.random_normal_initializer(stddev=0.02))
		b = tf.get_variable("bias",[filter_num],initializer=tf.constant_initializer(0.0))
		conv = tf.nn.conv3d(x,W,strides=[1,1,1,1,1],padding='SAME')
		return tf.nn.relu(conv+b)

def conv_3d_linear(x,name,filter_num,filter_size):
	with tf.variable_scope(name):
		input_channel = x.get_shape()[4]
		W = tf.get_variable("weights",[filter_size[0],filter_size[1],filter_size[2],
							input_channel,filter_num],
							initializer=tf.random_normal_initializer(stddev=0.02))
		b = tf.get_variable("bias",[filter_num],initializer=tf.constant_initializer(0.0))
		conv = tf.nn.conv3d(x,W,strides=[1,1,1,1,1],padding='SAME')
		return (conv+b)

def residual_block(x,name,filter_size):
	with tf.variable_scope(name):
		filter_num = x.get_shape()[4]
		conv_1 = conv_3d(x,"conv_1",filter_num,filter_size)
		conv_2 = conv_3d_linear(conv_1,"conv_2",filter_num,filter_size)
		return tf.nn.relu(tf.add(x,conv_2))

def residual_block_3(x,name,filter_size):
	with tf.variable_scope(name):
		filter_num = x.get_shape()[4]
		conv_1 = conv_3d(x,"conv_1",filter_num,filter_size)
		conv_2 = conv_3d(conv_1,"conv_2",filter_num,filter_size)
		conv_3 = conv_3d_linear(conv_2,"conv_3",filter_num,filter_size)
		return tf.nn.relu(tf.add(x,conv_3))

def up_residual_block(x,y,name,filter_size):
	with tf.variable_scope(name):
		filter_num = x.get_shape()[4]
		xy = tf.concat([x,y],axis=4)
		conv_1 = conv_3d(xy,"conv_1",filter_num,filter_size)
		conv_2 = conv_3d_linear(conv_1,"conv_2",filter_num,filter_size)
		return tf.nn.relu(tf.add(x,conv_2))

def up_residual_block_3(x,y,name,filter_size):
	with tf.variable_scope(name):
		filter_num = x.get_shape()[4]
		xy = tf.concat([x,y],axis=4)
		conv_1 = conv_3d(xy,"conv_1",filter_num,filter_size)
		conv_2 = conv_3d(conv_1,"conv_2",filter_num,filter_size)
		conv_3 = conv_3d_linear(conv_2,"conv_3",filter_num,filter_size)
		return tf.nn.relu(tf.add(x,conv_3))

def conv_3d_sigmoid(x,name,filter_num,filter_size):
	with tf.variable_scope(name):
		input_channel = x.get_shape()[4]
		W = tf.get_variable("weights",[filter_size[0],filter_size[1],filter_size[2],
							input_channel,filter_num],
							initializer=tf.random_normal_initializer(stddev=0.02))
		b = tf.get_variable("bias",[filter_num],initializer=tf.constant_initializer(0.0))
		conv = tf.nn.conv3d(x,W,strides=[1,1,1,1,1],padding='SAME')
		return tf.sigmoid(conv+b)

def deconv_3d(x,name,filter_num,filter_size,strides):
	with tf.variable_scope(name):
		x_shape = x.get_shape().as_list()
		input_channel = x_shape[4]
		output_shape = [x_shape[0],x_shape[1]*strides[0],x_shape[2]*strides[1],
						x_shape[3]*strides[2],filter_num]
		W = tf.get_variable("weights",[filter_size[0],filter_size[1],filter_size[2],
							filter_num,input_channel],
							initializer=tf.random_normal_initializer(stddev=0.02))
		b = tf.get_variable("bias",[filter_num],initializer=tf.constant_initializer(0.0))
		conv = tf.nn.conv3d_transpose(x,W,output_shape=output_shape,strides=[1,strides[0],strides[1],strides[2],1],padding='SAME')
		return tf.nn.relu(conv+b)

def get_v_net(x,name="u_net",b=1):
	with tf.variable_scope(name):
		conv1 = conv_3d(x,"conv1_1",b*16,[3,3,3])
		conv1 = conv_3d(conv1,"conv1_2",b*16,[3,3,3])
		pool1 = conv_3d_pool(conv1,"pool1",b*32,[3,3,3],[2,2,2])

		conv2 = residual_block(pool1,"conv2_1",[3,3,3])
		pool2 = conv_3d_pool(conv2,"pool2",b*64,[3,3,3],[2,2,2])

		conv3 = residual_block_3(pool2,"conv3_1",[3,3,3])
		pool3 = conv_3d_pool(conv3,"pool3",b*128,[3,3,3],[2,2,2])

		conv4 = residual_block_3(pool3,"conv4_1",[3,3,3])

		up5 = deconv_3d(conv4,"deconv5",b*64,(2,2,2),(2,2,2))
		conv5 = up_residual_block_3(up5,conv3,"conv5_1",[3,3,3])

		up6 = deconv_3d(conv5,"deconv6",b*32,(2,2,2),(2,2,2))
		conv6 = up_residual_block(up6,conv2,"conv6_1",[3,3,3])

		up7 = deconv_3d(conv6,"deconv7",b*16,(2,2,2),(2,2,2))
		conv7 = up_residual_block(up7,conv1,"conv7_1",[3,3,3])

		#conv8 = conv_3d_linear(conv7,"conv8",2,[1,1,1])
		conv8 = conv_3d_sigmoid(conv7,"conv8",1,[1,1,1])
		return conv8

'''def dice_loss(conv_out,y,name="dice_loss"):
	with tf.variable_scope(name):
		y_true = tf.contrib.layers.flatten(tf.cast(y, tf.float32))
		y_pred = tf.contrib.layers.flatten(conv_out)
		intersection = tf.reduce_sum(y_true * y_pred,1)
		#dc = (2. * intersection + 0.01) / (tf.reduce_sum(y_true,1) + tf.reduce_sum(y_pred,1) + 0.01)
		dc = (2. * intersection + 0.01) / (tf.reduce_sum(y_true*y_true,1) + tf.reduce_sum(y_pred*y_pred,1) + 0.01)
		dc_mean = tf.reduce_mean(dc)
		dc_loss = -dc_mean
		return dc_loss'''
def dice_loss(y1,y2):
	y_true = tf.contrib.layers.flatten(y2)
	y_pred = tf.contrib.layers.flatten(y1)
	intersection = tf.reduce_sum(y_true * y_pred,1)
	#dc = (2. * intersection + 0.01) / (tf.reduce_sum(y_true*y_true,1) + tf.reduce_sum(y_pred*y_pred,1) + 0.01)
	dc = (2. * intersection + 0.01) / (tf.reduce_sum(y_true,1) + tf.reduce_sum(y_pred,1) + 0.01)
	dc_loss = - tf.reduce_mean(dc)
	return dc_loss

def dice(y1,y2):
	y_true = tf.contrib.layers.flatten(y2)
	y_pred = (tf.sign(tf.contrib.layers.flatten(y1) - 0.5) + 1)/2.0
	intersection = tf.reduce_sum(y_true * y_pred,1)
	dc = (2. * intersection + 0.01) / (tf.reduce_sum(y_true,1) + tf.reduce_sum(y_pred,1) + 0.01)
	dc_coef = tf.reduce_mean(dc)
	return dc_coef

def get_train_op(loss,lr=0.0001):
	optimizer =  tf.train.AdamOptimizer(lr)
	train_op = optimizer.minimize(loss)
	return train_op

'''
TODO:
Weighted cross entropy loss
Learning rate decay
'''

def weighted_softmax_loss(y_true, y_pred_linear, pos_weight):
	y_true_one_hot = tf.one_hot(y_true, depth=2)
	flat_y_true = tf.reshape(y_true_one_hot, [-1, 2])
	flat_y_pred = tf.reshape(y_pred_linear, [-1, 2])
	weights = tf.constant(np.array([1.0, pos_weight], dtype=np.float32))
	case_weights = tf.reduce_sum(tf.multiply(flat_y_true, weights),axis=1)
	loss = tf.nn.softmax_cross_entropy_with_logits(labels=flat_y_true,logits=flat_y_pred)
	ave_loss = tf.reduce_mean(tf.multiply(loss, case_weights))
	return ave_loss

def pred_prob(y_pred_linear):
	return (tf.nn.softmax(y_pred_linear))[:,:,:,:,1]
