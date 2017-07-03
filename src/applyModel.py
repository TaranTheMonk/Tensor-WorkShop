import numpy as np
import json
import tensorflow as tf
import os
import pandas as pd

##ALB, BET, DOL, LAG, NoF, OTHER, SHARK, YFT

def processRawData(filePath: str) -> np.array:
    with open(filePath, 'r', encoding='utf-8') as f:
        rawData = json.load(f)
    f.close()
    nameArray = list()
    valueArray = list()
    for key in rawData.keys():
        nameArray.append([key])
        valueArray.append(rawData[key])
    valueArray = np.array(valueArray, dtype='float32')
    nameArray = np.array(nameArray)
    return valueArray, nameArray

def cnnModel(xTest, checkPointPath):
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        # ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor
        # strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ##set input
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

    ##conv layer 1 and pool layer 1
    W_conv1 = weight_variable([3, 3, 3, 32], name='W_conv1')
    b_conv1 = bias_variable([32], name='b_conv1')
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    ##conv layer 2 and pool layer 2
    W_conv2 = weight_variable([3, 3, 32, 64], name='W_conv2')
    b_conv2 =bias_variable([64], name='b_conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    ##conv layer 3 and pool layer 3
    W_conv3 = weight_variable([3, 3, 64, 128], name='W_conv3')
    b_conv3 =bias_variable([128], name='b_conv3')
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    ##FC layer 1
    W_fc1 = weight_variable((8 * 8 * 128, 1024), name='W_fc1')
    b_fc1 = bias_variable([1024], name='b_fc1')
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    ##dropout to avoid overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ##readout layer
    W_fc2 = weight_variable([1024, 8], name='W_fc2')
    b_fc2 = bias_variable([8], name='b_fc2')
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    saver = tf.train.Saver()
    saver.restore(sess, checkPointPath)
    print('Model restored')

    result = sess.run(y_conv, feed_dict={x: xTest, keep_prob: 1.0})
    y_conv_result = sess.run(tf.nn.softmax(result))
    return y_conv_result

xPath = os.path.expanduser('~/Desktop/Projects/Tensor-WorkShop/data/X-Test-Tensor.json')
xArray, xName = processRawData(xPath)
xArray = xArray
xName = xName

checkPointPath = os.path.expanduser('~/Desktop/Projects/Tensor-WorkShop/src/parameters/model-v1.0.ckpt')

result = cnnModel(xArray, checkPointPath)

output = np.concatenate((xName, result), axis=1)

headers = ['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
df = pd.DataFrame(output)
df.columns = headers
df.to_csv('submission_stg2.csv', index=False)