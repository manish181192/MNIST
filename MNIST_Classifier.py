from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import sys
import os
from LSTM import rnn_network
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
    :argument  /home/... LSTM 3 0.5
    [] => Model {'LSTM', 'gru', 'cnn'}
    [] => num_layers
    [] => Dropout Keep prob
'''
num_classes = 10

save_path = ("_").join(sys.argv)
save_path = save_path+"/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    print "Directory already exists..!!! "
    exit(0)

def get_network():
    model = sys.argv[1]
    num_lstm_layers = int(sys.argv[2])
    dropout = float(sys.argv[3])
    network = rnn_network(type=model,
                          inputs=x,
                          num_classes=num_classes,
                          dropout=dropout,
                          num_lstm_layers=num_lstm_layers)
    return network

x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

rnn_network = get_network()
logits = rnn_network.logits

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

indices = tf.argmax(logits, axis=1)
prediction = tf.one_hot(indices= indices,
                        depth=num_classes,
                        on_value=1.0,
                        off_value=0.0,
                        axis=-1)
total_correct = tf.reduce_sum(tf.mul(y,prediction))


print 'Graph done'

opt = tf.train.GradientDescentOptimizer(learning_rate= 0.1)
grads_vars = opt.compute_gradients(loss)
train_op = opt.apply_gradients(grads_and_vars=grads_vars)

saver = tf.train.Saver()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    max_test_acc = -1
    for epoch in range(10000):
        x_, y_batch = mnist.train.next_batch(100)
        x_batch = np.reshape(x_, newshape=[-1, 28,28])
        _, loss_ = session.run([train_op, loss], feed_dict={x:x_batch,y:y_batch})
        print 'Epoch'+str(epoch)+ ' loss: '+str(loss_)
        if epoch %99 == 0:

            x_test = np.reshape(mnist.test.images, [-1,28,28])
            y_test = mnist.test.labels
            num = len(y_test)
            t_corr, loss_test = session.run([total_correct, loss], feed_dict={x:x_test, y:y_test})
            acc = t_corr/num
            print 'Test Accuracy'+str(acc)+'   loss:'+str(loss_test)
            if acc>max_test_acc:
                max_test_acc = acc
                print saver.save(sess=session, save_path=save_path)
            print 'Max accuracy : '+str(max_test_acc)