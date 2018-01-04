import tensorflow as tf
import numpy as np
import os
import time
import datetime
import scipy.misc
from load_data import LoadTrainBatch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

[x,y] = LoadTrainBatch(1)
def conv_transpose(x,conv,stride,pre_layer):
    shape = (conv,conv,1,1)
    filter = tf.constant(1.0,shape=shape)
    strides = (1,stride,stride,1)
    out_shape = [1] + pre_layer.get_shape().as_list()[1:]
    return tf.nn.conv2d_transpose(x, filter = filter, output_shape = out_shape, strides = strides, padding='VALID')

def get_mean(x):
    x = tf.reduce_mean(x, axis= -1)
    x = tf.expand_dims(x, axis = -1)
    return x

def normalize(x):
    normed= tf.div(tf.subtract(x,
                               tf.reduce_min(x)),
                   tf.subtract(tf.reduce_max(x),
                               tf.reduce_min(x))
                   )
    return normed

with tf.device('/gpu:0'):
    graph = tf.get_default_graph()
    config = tf.ConfigProto(
      allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        ckpt = tf.train.get_checkpoint_state('./save/')
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, ckpt.model_checkpoint_path)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        conv1 = graph.get_operation_by_name("conv_layer/conv1").outputs[0]
        conv2 = graph.get_operation_by_name("conv_layer/conv2").outputs[0]
        conv3 = graph.get_operation_by_name("conv_layer/conv3").outputs[0]
        conv4 = graph.get_operation_by_name("conv_layer/conv4").outputs[0]
        conv5 = graph.get_operation_by_name("conv_layer/conv5").outputs[0]

        conv5_mean = get_mean(conv5)
        conv4_mean = get_mean(conv4)
        conv3_mean = get_mean(conv3)
        conv2_mean = get_mean(conv2)
        conv1_mean = get_mean(conv1)

        input_mean = get_mean(input_x)

        conv5_t = conv_transpose(conv5_mean,3,1,conv4_mean)

        temp5 = tf.multiply(conv5_t,conv4_mean)
        conv4_t = conv_transpose(temp5,3,1,conv3_mean)

        temp4 = tf.multiply(conv4_t, conv3_mean)
        conv3_t = conv_transpose(temp4, 5, 2, conv2_mean)

        temp3 = tf.multiply(conv3_t, conv2_mean)
        conv2_t = conv_transpose(temp3, 5, 2, conv1_mean)

        temp2 = tf.multiply(conv2_t, conv1_mean)
        conv1_t = conv_transpose(temp2, 5, 2, input_mean)
        conv1_t = normalize(conv1_t)

        mask5,mask4,mask3,mask2,mask1 = sess.run([conv5_t,conv4_t,conv3_t,conv2_t,conv1_t],feed_dict={input_x:x})
        conv5_val,conv4_val,conv3_val,conv2_val,conv1_val = sess.run([conv5_mean,conv4_mean,conv3_mean,conv2_mean,
                                                                      conv1_mean],feed_dict={input_x:x})

        fig = plt.figure(figsize=(25,20))
        ax1 = fig.add_subplot(5, 2, 1)
        ax1.imshow(mask1.reshape(mask1.shape[1],mask1.shape[2]), cmap=plt.cm.gray)
        ax1 = fig.add_subplot(5, 2, 2)
        ax1.imshow(conv1_val.reshape(conv1_val.shape[1],conv1_val.shape[2]), cmap=plt.cm.gray)
        ax2 = fig.add_subplot(5, 2, 3)
        ax2.imshow(mask2.reshape(mask2.shape[1],mask2.shape[2]), cmap=plt.cm.gray)
        ax2 = fig.add_subplot(5, 2, 4)
        ax2.imshow(conv2_val.reshape(conv2_val.shape[1],conv2_val.shape[2]), cmap=plt.cm.gray)
        ax3 = fig.add_subplot(5, 2, 5)
        ax3.imshow(mask3.reshape(mask3.shape[1],mask3.shape[2]), cmap=plt.cm.gray)
        ax3 = fig.add_subplot(5, 2, 6)
        ax3.imshow(conv3_val.reshape(conv3_val.shape[1],conv3_val.shape[2]), cmap=plt.cm.gray)
        ax4 = fig.add_subplot(5, 2, 7)
        ax4.imshow(mask4.reshape(mask4.shape[1],mask4.shape[2]), cmap=plt.cm.gray)
        ax4 = fig.add_subplot(5, 2, 8)
        ax4.imshow(conv4_val.reshape(conv4_val.shape[1],conv4_val.shape[2]), cmap=plt.cm.gray)
        ax5 = fig.add_subplot(5, 2, 9)
        ax5.imshow(mask5.reshape(mask5.shape[1],mask5.shape[2]), cmap=plt.cm.gray)
        ax5 = fig.add_subplot(5, 2, 10)
        ax5.imshow(conv5_val.reshape(conv5_val.shape[1],conv5_val.shape[2]), cmap=plt.cm.gray)

        plt.savefig('rlst.jpg')




