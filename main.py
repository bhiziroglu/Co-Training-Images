import tensorflow as tf
import numpy as np
import math
import platform
import cPickle as pickle
import random
import os
import cifar


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
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    return X_train, y_train, X_val, y_val, X_test, y_test




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


'''Stratifying the labeled dataset'''
'''1000 Images -> 100 Images of each class'''
L = []   # a set L of labeled training examples (1000, 32, 32, 3)
L_y = [] # (1000,)
U = []   # a set U of unlabeled examples (48000, 32, 32, 3)

positive_counter = 0
negative_counter = 0
seen_example_indices = [] # Store the seen examples and remove them from the unlabeled dataset
LABELED_DATA_SET_SIZE = 4000
for index in range(len(X_train)):

    if positive_counter == LABELED_DATA_SET_SIZE/10 and negative_counter == (LABELED_DATA_SET_SIZE - LABELED_DATA_SET_SIZE/10):
        break

    if y_train[index] == 1:
        if positive_counter == LABELED_DATA_SET_SIZE/10:
            continue
        positive_counter += 1
        L.append(X_train[index])
        L_y.append(1)
        seen_example_indices.append(index)
    else:
        if negative_counter == (LABELED_DATA_SET_SIZE - LABELED_DATA_SET_SIZE/10):
            continue
        negative_counter += 1
        L.append(X_train[index])
        L_y.append(0)
        seen_example_indices.append(index)

# Unlabeled dataset formed by examples that is not in the labeled dataset
Ucounter = 0
cnt = 0
while Ucounter != (48000 - LABELED_DATA_SET_SIZE):

    if cnt in seen_example_indices:
        cnt = cnt + 1
    else:
        Ucounter += 1
        U.append(X_train[cnt])
        cnt = cnt + 1

L = np.asarray(L) # a set L of labeled training examples (1000, 32, 32, 3)
L_y = np.asarray(L_y) 
U = np.asarray(U, dtype = np.int8) # a set U of unlabeled examples (48000, 32, 32, 3)

tf.reset_default_graph()

#with tf.variable_scope('h1'):
X1 = tf.placeholder(tf.float32, [None, 32, 16, 3])
y1 = tf.placeholder(tf.int64, [None])
is_training1 = tf.placeholder(tf.bool)
h1 = cifar.CifarNet(X1,y1,is_training1,"1")
h1.forward()
h1.set_params()

#with tf.variable_scope('h2'):
X2 = tf.placeholder(tf.float32, [None, 32, 16, 3])
y2 = tf.placeholder(tf.int64, [None])
is_training2 = tf.placeholder(tf.bool)
h2 = cifar.CifarNet(X2,y2,is_training2,"2")
h2.forward()
h2.set_params()

# train with 10 epochs
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


u = 1000 # Choose u examples from U
Uhat = []
tmp = list(U)
for i in range(u):
    Uhat.append(tmp.pop())
del tmp

# Create a pool U' of examples by choosing u examples random from U
# U' created (Uhat)
P = 100
N = 100

for k in range(500): 

    # Use L to train a classifier h1 that considers only the x1 portion of x
    #print('Training h1')
    h1.run(sess, L[:,:,0:16,:], L_y, k, 1, 64, 200, plot_losses=False)

    # Use L to train a classifier h2 that considers only the x2 portion of x
    #print('Training h2')
    h2.run(sess, L[:,:,16:,:], L_y, k, 1, 64, 200, plot_losses=False)

    # Allow h1 to label p positive and n negative examples from U'
    i = 0
    np.random.shuffle(Uhat)
    Uhat = list(Uhat)
    
    positives = []
    negatives = []
    p = 1 * P
    n = 1 * N
    while(i < len(Uhat)):
        if p<1 and n<1:
            break

        ex = np.asarray(Uhat[i]) # (32,32,3)
        ex = np.reshape(ex,(1,32,32,3)) # (1,32,32,3)
        pred1 = h1.infer(sess,ex[:,:,0:16,:])
        pred1 = np.argmax(pred1)

        if pred1 == 1 and p>0:
            positives.append(np.reshape(ex,(32,32,3))) # To preserve the shape after append!
            Uhat.pop(i) # Remove that example from U'
            p -= 1
        elif pred1 == 0 and n>0:
            negatives.append(np.reshape(ex,(32,32,3)))
            Uhat.pop(i)
            n -= 1

        i += 1

    # Adding self-labeled examples to L
    L = list(L)
    L_y = list(L_y)
    for p in positives:
        L.append(p)
        L_y.append(1) # Positive examples have a label of 1
    
    for n in negatives:
        L.append(n)
        L_y.append(0)

    # Allow h2 to label p positive and n negative examples from U'
    i = 0
    np.random.shuffle(Uhat)
    Uhat = list(Uhat)

    positives = []
    negatives = []
    p = 1 * P
    n = 1 * N
    while(i < len(Uhat)):
        if p<1 and n<1:
            break

        ex = np.asarray(Uhat[i]) # (32,32,3)
        ex = np.reshape(ex,(1,32,32,3)) # (1,32,32,3)
        pred2 = h2.infer(sess,ex[:,:,16:,:])
        pred2 = np.argmax(pred2)

        if pred2 == 1 and p>0:
            positives.append(np.reshape(ex,(32,32,3)))
            Uhat.pop(i) # Remove that example from U'
            p -= 1
        elif pred2 == 0 and n>0:
            negatives.append(np.reshape(ex,(32,32,3)))  
            Uhat.pop(i)
            n -= 1

        i += 1

    # Adding self-labeled examples to L
    for p in positives:
        L.append(p)
        L_y.append(1) # Positive examples have a label of 1
    
    for n in negatives:
        L.append(n)
        L_y.append(0)

    L = np.asarray(L)
    L_y = np.asarray(L_y)


    # Randomly choose 2p + 2n examples from U to replenish U'
    U = list(U)
    if (2*P + 2*N) > len(U): # Break if we run out of images in our unlabeled dataset
        break

    for i in range(2*P + 2*N):
        ex = np.asarray(U.pop()) #(32,32,3)
        Uhat.append(ex)


    U = np.asarray(U)
    
    h1.accuracy(sess,X_val[:,:,0:16,:],y_val)

    h2.accuracy(sess,X_val[:,:,16:,:],y_val)
