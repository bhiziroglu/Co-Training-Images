import tensorflow as tf
import numpy as np
import math

# Inspired from kaapex's Kaggle Kernel

class CifarNet():

    def __init__(self, X, y, is_training, modelName):
        self.X = X
        self.y = y
        self.is_training = is_training
        self.modelName = modelName

        self.Wconv1 = tf.get_variable("Wconv1"+modelName, shape=[5, 5, 3, 32])
        self.bconv1 = tf.get_variable("bconv1"+modelName, shape=[32])

        self.Wconv2 = tf.get_variable("Wconv2"+modelName, shape=[5, 5, 32, 64])
        self.bconv2 = tf.get_variable("bconv2"+modelName, shape=[64])

        self.W1 = tf.get_variable("W1"+modelName, shape=[1344, 1024])
        self.b1 = tf.get_variable("b1"+modelName, shape=[1024])

        self.W2 = tf.get_variable("W2"+modelName, shape=[1024, 2])
        self.b2 = tf.get_variable("b2"+modelName, shape=[2])         
        

    def set_params(self):

        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate = 1e-3
        self.end_learning_rate = 7e-3
        self.decay_steps = 10000

        self.learning_rate = tf.train.polynomial_decay(self.starter_learning_rate, self.global_step,
                                                self.decay_steps, self.end_learning_rate,
                                                power=0.5)

        self.exp_learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                    100000, 0.96, staircase=True)

        # define our loss
        self.cross_entr_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.y,2), logits=self.predict)
        self.mean_loss = tf.reduce_mean(self.cross_entr_loss)

        # define our optimizer
        self.optimizer = tf.train.AdamOptimizer(self.exp_learning_rate)

        self.train_step = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

    def forward(self):

        # define our graph (e.g. two_layer_convnet) with stride 1
        conv1 = tf.nn.conv2d(self.X, self.Wconv1, strides=[1, 1, 1, 1], padding='SAME') + self.bconv1
        # ReLU Activation Layer
        relu1 = tf.nn.relu(conv1)
        # Conv
        conv2 = tf.nn.conv2d(relu1, self.Wconv2, strides=[1, 2, 2, 1], padding='VALID') + self.bconv2
        # ReLU Activation Layer
        relu2 = tf.nn.relu(conv2)
        # 2x2 Max Pooling layer with a stride of 2
        maxpool = tf.layers.max_pooling2d(relu2, pool_size=(2,2), strides=2)
        maxpool_flat = tf.reshape(maxpool,[-1,1344])
        # Spatial Batch Normalization Layer (trainable parameters, with scale and centering)
        bn1 = tf.layers.batch_normalization(inputs=maxpool_flat, center=True, scale=True, training=self.is_training)
        # Affine layer with 1024 output units
        affine1 = tf.matmul(bn1, self.W1) + self.b1
        # vanilla batch normalization
        affine1_flat = tf.reshape(affine1,[-1,1024])
        bn2 = tf.layers.batch_normalization(inputs=affine1, center=True, scale=True, training=self.is_training)
        # ReLU Activation Layer
        relu2 = tf.nn.relu(bn2)
        # dropout
        drop1 = tf.layers.dropout(inputs=relu2, training=self.is_training)
        # Affine layer from 1024 input units to 10 outputs
        affine2 = tf.matmul(drop1, self.W2) + self.b2
        # vanilla batch normalization
        self.predict = tf.layers.batch_normalization(inputs=affine2, center=True, scale=True, training=self.is_training)
        return self.predict

    def infer(self, session, Xd):
        prediction = self.predict
        variables = [prediction]
        
        feed_dict = {self.X: Xd,
                    self.is_training: False }

        pred = session.run(variables,feed_dict=feed_dict)
        
        return pred

    def accuracy(self, session, Xd, yd):
        yd = np.reshape(yd,(1000)) # Shape (1000,)

        preds = self.infer(session,Xd)
        res = np.argmax(preds,2)
        res = np.reshape(res,(1000)) # Shape (1000,)

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for index in range(len(res)):

            if res[index] == 1 and yd[index] == 1:
                true_positives += 1.0
            elif res[index] == 0 and yd[index] == 0:
                true_negatives += 1.0
            elif res[index] == 1 and yd[index] == 0:
                false_positives += 1.0
            elif res[index] == 0 and yd[index] == 1:
                false_negatives += 1.0

        accuracy = (true_positives + true_negatives) / (true_negatives + true_positives + false_negatives + false_positives)
        if true_positives == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1 = (2 * recall * precision) / (recall + precision)

        print(self.modelName + ": Accuracy: ",accuracy)
        print(self.modelName + ": Precision: ",precision)
        print(self.modelName + ": Recall: ",recall)
        print(self.modelName + ": f1: ",f1)


    def run(self, session, Xd, yd, epoch_counter,
                  epochs=1, batch_size=64, print_every=100, 
                  plot_losses=False, isSoftMax=False):
        # have tensorflow compute accuracy
        if isSoftMax:
            correct_prediction = tf.nn.softmax(self.predict)
        else:
            correct_prediction = tf.equal(tf.argmax(self.predict,1), self.y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = self.train_step is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [self.mean_loss, correct_prediction, accuracy]
        if self.train_step:
            variables[-1] = self.train_step

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
                feed_dict = {self.X: Xd[idx,:],
                             self.y: yd[idx],
                             self.is_training: training_now }
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
                    print("Model:"+self.modelName + " Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                          .format(iter_cnt,loss,np.sum(corr)*1.0/actual_batch_size))
                iter_cnt += 1
            total_correct = correct*1.0/Xd.shape[0]
            total_loss = np.sum(losses)*1.0/Xd.shape[0]
            s2 = self.modelName + ": " + str(total_loss) + "\n"
            f3 = open("loss.txt", "a")
            f3.write(s2)
            f3.close()

            s_train_accuracy = self.modelName + ": " + str(total_correct) + "\n"
            f5 = open("train_acc.txt","a")
            f5.write(s_train_accuracy)
            f5.close()

            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                  .format(total_loss,total_correct,epoch_counter+1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        return total_loss, total_correct
