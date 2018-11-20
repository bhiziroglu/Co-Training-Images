import tensorflow as tf
import numpy as np
import math
import platform
import cPickle as pickle
import random
import os
import cifar

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#y_train
#array([6, 9, 9, ..., 4, 9, 3])

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

    
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    # Load the raw CIFAR-10 data
    cifar10_dir = './cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    # SUBTRACTING MEAN INCREASE ACCURACY ???? TRY OUT ???
    # Normalize the data: subtract the mean image # SUBTRACTING MEAN INCREASE ACCURACY ???? TRY OUT ???
    #mean_image = np.mean(X_train, axis=0)
    #X_train -= mean_image
    #X_val -= mean_image
    #X_test -= mean_image
    return X_train, y_train, X_val, y_val, X_test, y_test



# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

for i in range(len(y_train)):
    if y_train[i] == 1:
        continue
    else:
        y_train[i] = 0

for i in range(len(y_val)):
    if y_val[i] == 1:
        continue
    else:
        y_val[i] = 0

for i in range(len(y_test)):
    if y_test[i] == 1:
        continue
    else:
        y_test[i] = 0

L = X_train[0:1000,:,:,:] # a set L of labeled training examples
U = X_train[1000:,:,:,:] # a set U of unlabeled examples
L_y = y_train[0:1000]



tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
h1 = cifar.CifarNet(X,y,is_training)
h1.forward(X,y,is_training)
h1.set_params()


# train with 10 epochs
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for k in range(30):
    se = L[0:1,:,:,:]
    print('Predicting')
    p = h1.infer(sess,se)
    print('Prediction => ',np.argmax(p))
    print('Training')
    h1.run(sess, h1.mean_loss, L, L_y, 5, 64, 200, h1.train_step, plot_losses=False)
    print('Validation')
    h1.run(sess, h1.mean_loss, X_val, y_val, 1, 64)