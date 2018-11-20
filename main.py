import tensorflow as tf
import numpy as np
import math
import platform
import cPickle as pickle
import random
import os

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

# define net
class CifarNet():

    def __init__(self):
        # conv layer
        # H2 = (H1 - F + 2P)/S +1
        # (32-5)/1 + 1 = 28
        # 28x28x32 = 25088
        # To ReLu (?x16x16x32) -> MaxPool (?x16x16x32) -> affine (8192)
        self.Wconv1 = tf.get_variable("Wconv1", shape=[5, 5, 3, 32])
        self.bconv1 = tf.get_variable("bconv1", shape=[32])
        # (32-5)/1 + 1 = 28
        # 28x28x64 = 50176
        self.Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 32, 64])
        self.bconv2 = tf.get_variable("bconv2", shape=[64])
        # affine layer with 1024
        self.W1 = tf.get_variable("W1", shape=[3136, 1024])
        self.b1 = tf.get_variable("b1", shape=[1024])
        # affine layer with 10
        #self.W2 = tf.get_variable("W2", shape=[1024, 10])
        #self.b2 = tf.get_variable("b2", shape=[10])
        self.W2 = tf.get_variable("W2", shape=[1024, 2])
        self.b2 = tf.get_variable("b2", shape=[2])           
        

    def forward(self, X, y, is_training):
        # conv2d
        # ReLu
        # conv2d
        # ReLu
        # maxpool
        # Batch Norm
        # Affine
        # Batch Norm
        # ReLu
        # Affine
        # dropout
        # Batch Norm

        # conv layer
        # H2 = (H1 - F + 2P)/S +1
        # (32-5)/1 + 1 = 28
        # 28x28x32 = 25088
        # To ReLu (?x16x16x32) -> MaxPool (?x16x16x32) -> affine (8192)

        # define our graph (e.g. two_layer_convnet) with stride 1
        conv1 = tf.nn.conv2d(X, self.Wconv1, strides=[1, 1, 1, 1], padding='SAME') + self.bconv1
        print(conv1.shape)
        # ReLU Activation Layer
        relu1 = tf.nn.relu(conv1)
        print(relu1)
        # Conv
        conv2 = tf.nn.conv2d(relu1, self.Wconv2, strides=[1, 2, 2, 1], padding='VALID') + self.bconv2
        print(conv2.shape)
        # ReLU Activation Layer
        relu2 = tf.nn.relu(conv2)
        print(relu2)
        # 2x2 Max Pooling layer with a stride of 2
        maxpool = tf.layers.max_pooling2d(relu2, pool_size=(2,2), strides=2)
        print(maxpool.shape)
        maxpool_flat = tf.reshape(maxpool,[-1,3136])
        # Spatial Batch Normalization Layer (trainable parameters, with scale and centering)
        bn1 = tf.layers.batch_normalization(inputs=maxpool_flat, center=True, scale=True, training=is_training)
        # Affine layer with 1024 output units
        affine1 = tf.matmul(bn1, self.W1) + self.b1
        print(affine1.shape)
        # vanilla batch normalization
        affine1_flat = tf.reshape(affine1,[-1,1024])
        bn2 = tf.layers.batch_normalization(inputs=affine1, center=True, scale=True, training=is_training)
        print(bn2.shape)
        # ReLU Activation Layer
        relu2 = tf.nn.relu(bn2)
        print(relu2.shape)
        # dropout
        drop1 = tf.layers.dropout(inputs=relu2, training=is_training)
        # Affine layer from 1024 input units to 10 outputs
        affine2 = tf.matmul(drop1, self.W2) + self.b2
        # vanilla batch normalization
        self.predict = tf.layers.batch_normalization(inputs=affine2, center=True, scale=True, training=is_training)
        print(self.predict.shape)
        return self.predict

    def infer(self, session, Xd):
        prediction = self.predict
        variables = [prediction]
        
        feed_dict = {X: Xd,
                    is_training: False }

        pred = session.run(variables,feed_dict=feed_dict)
        
        return pred



    def run(self, session, loss_val, Xd, yd,
                  epochs=1, batch_size=64, print_every=100,
                  training=None, plot_losses=False, isSoftMax=False):
        # have tensorflow compute accuracy
        if isSoftMax:
            correct_prediction = tf.nn.softmax(self.predict)
        else:
            correct_prediction = tf.equal(tf.argmax(self.predict,1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [mean_loss, correct_prediction, accuracy]
        if training_now:
            variables[-1] = training

        # counter 
        iter_cnt = 0
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                # generate indicies for the batch
                start_idx = (i*batch_size)%Xd.shape[0]
                idx = train_indicies[start_idx:start_idx+batch_size]

                # create a feed dictionary for this batch
                feed_dict = {X: Xd[idx,:],
                             y: yd[idx],
                             is_training: training_now }
                # get batch size
                actual_batch_size = yd[idx].shape[0]

                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _ = session.run(variables,feed_dict=feed_dict)

                # aggregate performance stats
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)

                # print every now and then
                if training_now and (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                          .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                iter_cnt += 1
            total_correct = correct/Xd.shape[0]
            total_loss = np.sum(losses)/Xd.shape[0]
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                  .format(total_loss,total_correct,e+1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        return total_loss, total_correct


tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
h1 = CifarNet()
h1.forward(X,y,is_training)


global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
end_learning_rate = 5e-3
decay_steps = 10000

learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                        decay_steps, end_learning_rate,
                                        power=0.5)

exp_learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                            100000, 0.96, staircase=True)

# define our loss
cross_entr_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,2), logits=h1.predict)
mean_loss = tf.reduce_mean(cross_entr_loss)

# define our optimizer
optimizer = tf.train.AdamOptimizer(exp_learning_rate)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss, global_step=global_step)

# train with 10 epochs
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for k in range(30):
    se = L[0:1,:,:,:]
    print('Predicting')
    p = h1.infer(sess,se)
    print('Prediction => ',np.argmax(p))
    print('Training')
    h1.run(sess, mean_loss, L, L_y, 5, 64, 200, train_step, plot_losses=False)
    print('Validation')
    h1.run(sess, mean_loss, X_val, y_val, 1, 64)