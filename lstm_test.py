# In[1]

import tensorflow as tf
#import LSTM  as contrib_rnn_cell
#import NLSTM as contrib_rnn_cell
import NLSTM_1 as contrib_rnn_cell

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("D:/tmp/data/", one_hot=True)

learning_rate = 0.001
training_iters = 1000
batch_size = 128
display_step = 10

n_input = 28  
n_steps = 28  
n_hidden_ = 128  
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
weight= tf.Variable(tf.truncated_normal([n_hidden_, n_classes]))
biase = tf.Variable(tf.zeros([n_classes]))

def RNN(_X, weight, biase):

    #cell = contrib_rnn_cell.LSTMCell(n_hidden_)
    cell = contrib_rnn_cell.NLSTMCell(n_hidden=n_hidden_,depth=3)

    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, _ = tf.nn.dynamic_rnn(cell, _X, initial_state=init_state)
    

    logits=tf.matmul(outputs[:,-1,:], weight) + biase
    return logits

pred = RNN(x, weight, biase)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# In[6]

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step=0
    
    while step  < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        
        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print ("Iter " + str(step) + ", Minibatch Loss= " +          
                "{:.6f}".format(loss) + ", Training Accuracy= " +          
                "{:.5f}".format(acc))
        step+=1
    print ("Optimization Finished!")

# In[9]:

    test_len = batch_size
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
   
