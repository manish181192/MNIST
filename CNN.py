import tensorflow as tf
import numpy as np
import sys
import os
import time

class Convolve(object):

    def __init__(self, ip_shape, channels, filter_sizes, strides, padding, save_path):

        self.strides = strides
        self.padding = padding
        self.filter_sizes = filter_sizes
        self.channels = channels
        self.ip_shape = ip_shape
        self.N = self.ip_shape[0]
        self.save_path = save_path

    def graph_init(self):
        ip_ = tf.placeholder(dtype=tf.float32, shape=[None] + self.ip_shape + [self.channels])
        # self.channels = ip_shape[-1]
        self.keep_prob = tf.placeholder(dtype=tf.float32)

        # self.init_weights()
        with tf.variable_scope("CNN"):
            ip = ip_
            out_size = None
            for id, f_size in enumerate(self.filter_sizes):
                out_size = self.N - f_size + 1
                weight_filter = tf.get_variable("filter_weight_" + str(id), shape=[1, f_size, f_size, self.channels],
                                                initializer=tf.contrib.layers.xavier_initializer())
                bias_filter = tf.get_variable("filter_weight_" + str(id),
                                              initializer=tf.zeros_initializer([out_size, out_size, self.channels]))
                ip = tf.matmul(ip, weight_filter) + bias_filter

        conv_output = ip
        output_size = out_size
        return conv_output, output_size

    def get_optimization_function(self, logits):
        labels = tf.placeholder(dtype= tf.float32, shape= [None])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        correct_predictions = tf.equal(tf.argmax(labels,1), tf.argmax(logits,1))
        return loss , correct_predictions

    def train_step(self, learning_rate, loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        grads_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars=grads_vars)
        return train_op

    def conv(self, feed_dict):
        with tf.Session() as session:
            conv_output = self.graph_init()
            saver = tf.train.Saver()
            session.run(tf.global_variables_initializer())
            saver.restore(session, self.save_path)
            output = session.run([conv_output], feed_dict=feed_dict)
            session.close()
            return output

    def train_conv(self, learning_rate, feed_dict):
        with tf.Session() as session:
            conv_output = self.graph_init()
            loss, correct_prediction = self.get_optimization_function(conv_output)
            train_op = self.train_step(learning_rate= learning_rate, loss=loss)
            saver = tf.train.Saver()
            session.run(tf.global_variables_initializer())
            saver.restore(session, self.save_path)
            start_time = time.time()
            _, output, cor_pred = session.run([train_op, loss, correct_prediction], feed_dict=feed_dict)
            acc = cor_pred/len(feed_dict[0])
            print("Iter " + str(len(feed_dict[0])) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc) + "epoch Time: " + str(time.time() - start_time))
            session.close()
            return output
