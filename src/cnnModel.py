import tensorflow as tf
import numpy as np
import os
import json
from typing import *

def processRawData(filePath: str) -> np.array:
    with open(filePath, 'r', encoding='utf-8') as f:
        rawData = json.load(f)
    f.close()
    retArray = list()
    for key in rawData.keys():
        retArray.append(rawData[key])
    retArray = np.array(retArray, dtype='float32')
    return retArray

def dataPartition(originX: np.array, originY: np.array, splitRatio: Tuple[float, float]) \
        -> (np.array, np.array, np.array, np.array):
    partitionScale = originX.shape[0]
    trainSize = round(splitRatio[0] * partitionScale)
    indices = np.random.permutation(originX.shape[0])
    trainIndices = indices[:trainSize]
    testIndices = indices[trainSize:]
    trainArray_X = originX[trainIndices]
    testArray_X = originX[testIndices]
    trainArray_Y = originY[trainIndices]
    testArray_Y = originY[testIndices]
    return trainArray_X, testArray_X, trainArray_Y, testArray_Y

def cnnModel(xTrain, xTest, yTrain, yTest, numIterations, newModel=True, checkPointPath=None):

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
    y_ = tf.placeholder(tf.float32, shape=[None, 8])

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

    ##training model
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    if newModel == True:
        sess.run(tf.global_variables_initializer())
        print('Model initialized')
    else:
        saver.restore(sess, checkPointPath)
        print('Model restored')

    result = {'train': [], 'test': [], 'step': []}
    batchSize = 100
    for i in range(numIterations):
        indices = np.random.permutation(xTrain.shape[0])
        batchIndices = indices[batchSize:]
        print('step %d' % i)
        if i % 5 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={x: xTrain[batchIndices], y_: yTrain[batchIndices], keep_prob: 1.0})
            print('step %d, training accuracy: %g' % (i, train_accuracy))
            test_accuracy = accuracy.eval(
                feed_dict={x: xTest, y_: yTest, keep_prob: 1.0})
            print('step %d, test accuracy: %g' % (i, test_accuracy))
            result['train'].append(float(train_accuracy))
            result['test'].append(float(test_accuracy))
            result['step'].append(i)
        train_step.run(feed_dict={x: xTrain[batchIndices], y_: yTrain[batchIndices], keep_prob: 0.5})
    with open('resultRecord.json', 'w', encoding='utf-8') as f:
        json.dump(result, f)
    f.close()

    savePath = saver.save(sess, './parameters/model-v1.0.ckpt')
    print('Check point file successfully saved in %s' % savePath)

def main():
    ##load data from files
    xPath = os.path.expanduser('~/Desktop/Projects/Tensor-WorkShop/data/X-Tensor.json')
    yPath = os.path.expanduser('~/Desktop/Projects/Tensor-WorkShop/data/Y-Tensor.json')
    x = processRawData(xPath)
    y = processRawData(yPath)
    print('load data successfully')

    trainX, testX, trainY, testY = dataPartition(x, y, (0.8, 0.2))
    ##split data set into train and test
    print('split data successfully')

    print('Begin training...')
    cnnModel(trainX, testX, trainY, testY, numIterations=100, newModel=False, checkPointPath='./parameters/model-v1.0.ckpt')

if __name__ == '__main__':
    main()