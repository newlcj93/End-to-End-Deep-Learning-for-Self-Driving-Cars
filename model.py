'''
You are going to implement the CNN model from paper 'End to End Learning for Self-Driving Cars'.
Write the model below.
'''


import numpy as np
import tensorflow as tf
class Model(object):
    def __init__(self, lambda_l2 = 0.0001):
        self.x = tf.placeholder(tf.float32,shape = [None,66,200,3],name = "input_x")
        self.y_ = tf.placeholder(tf.float32,shape = [None,1],name = "input_y")
        self.keep_prob = tf.placeholder(tf.float32,name = "keep_prob")
        self.lambda_l2 = lambda_l2
        with tf.name_scope("conv_layer"):
            self.conv1 = self.conv_layer(self.x,conv = (5,5),stride=2,n_filters = 24,name = "conv1")
            self.conv2 = self.conv_layer(self.conv1,conv = (5,5),stride=2,n_filters = 36,name = "conv2")
            self.conv3 = self.conv_layer(self.conv2,conv = (5,5),stride=2,n_filters = 48,name = "conv3")
            self.conv4 = self.conv_layer(self.conv3,conv = (3,3),stride=1,n_filters = 64,name = "conv4")
            self.conv5 = self.conv_layer(self.conv4,conv = (3,3),stride=1,n_filters = 64,name = "conv5")
            self.flat = self.flatten(self.conv5,name = "flat")
        with tf.name_scope("fc_layer"):
            self.fc1 = self.fc_layer(self.flat,n_neurons=100,keep_prob = self.keep_prob,name = "fc1")
            self.fc2 = self.fc_layer(self.fc1,n_neurons=50,keep_prob = self.keep_prob,name = "fc2")
            self.fc3 = self.fc_layer(self.fc2,n_neurons=10,keep_prob = self.keep_prob,name = "fc3")
        self.y = self.output(self.fc3)
        self.loss = tf.reduce_mean(tf.square(tf.subtract(self.y_, self.y)),name = "loss")

    def weight_variable(self,shape,lambda_l2 = 0.001):
        initial = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lambda_l2)(initial))
        return initial

    def bias_variable(self,shape):
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x,W,stride):
        return tf.nn.conv2d(x,W,strides = [1,stride,stride,1],padding = 'VALID')

    def conv_layer(self,x,conv,stride=1,n_filters = 32,use_bias = True,name = None):
        W = self.weight_variable([conv[0],conv[1],x.get_shape()[-1].value,n_filters])
        if use_bias:
            b = self.bias_variable([n_filters])
            return tf.nn.relu(self.conv2d(x,W,stride = stride) + b,name= name)
        else:
            return tf.nn.relu(self.conv2d(x,W,stride = stride),name= name)

    def fc_layer(self, x,n_neurons,activation = tf.nn.relu,keep_prob = 1.0,name= None):
        W = self.weight_variable([x.get_shape()[-1].value,n_neurons])
        b = self.bias_variable([n_neurons])
        h = activation(tf.matmul(x,W) + b)
        h_drop = tf.nn.dropout(h,keep_prob,name = name)
        return h_drop

    def max_pool(self, x,ksize = (2,2),stride = 2):
        return tf.nn.max_pool(x,ksize = [1,ksize[0],ksize[1],1],strides = [1,stride,stride,1],padding = 'SAME')

    def flatten(self, x,name):
        product = 1
        for d in x.get_shape():
            if d.value is not None:
                product *= d.value
        return tf.reshape(x, [-1, product],name = name)

    def output(self, x):
        W = self.weight_variable([x.get_shape()[-1].value,1])
        b = self.bias_variable([1])
        return tf.nn.xw_plus_b(x,W,b,name = "output")

if __name__ == "__main__":
    model = Model()




