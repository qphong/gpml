import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp
 
import time 
import pickle

import sys 

import util 
import gp_util


class MAML():

    def __init__(self, 
            xdim, ydim,

            approximator, # the machine learning model
            task, # task to learn
            
            training_task_batchsize, # ndataset
            training_task_datasize, # npoint
            
            stepsize, # opt_step_size
            n_step, # n_stepsize

            dtype = tf.float32):

        self.xdim = xdim 
        self.ydim = ydim

        self.training_task_batchsize = training_task_batchsize
        self.training_task_datasize = training_task_datasize

        self.stepsize = stepsize
        self.n_step = n_step 
        

        self.approximator = approximator
        self.n_param = approximator.n_param
        
        self.task = task

        self.dtype = dtype 

        self.graph = tf.Graph()

        self.make_placeholders()
        self.make_variables()
        self.make_tensors()


    def make_placeholders(self):

        with self.graph.as_default():

            # Xtrain, Ytrain are randomly drawn from the distribution of tasks
            self.Xtrain_plc = tf.placeholder(
                                dtype=self.dtype, 
                                shape=(None,None,self.xdim), 
                                name='Xtrain_plc')
            # (training_task_batchsize, training_task_datasize, xdim)

            self.Ytrain_plc = tf.placeholder(
                                dtype=self.dtype, 
                                shape=(None,None,self.ydim), name='Ytrain_plc')
            # (training_task_batchsize, training_task_datasize, ydim)

            self.Xval_plc = tf.placeholder(
                                dtype=self.dtype, 
                                shape=(None,None,self.xdim), 
                                name='Xval_plc')
            # (val_task_batchsize, val_task_datasize, xdim)

            self.Yval_plc = tf.placeholder(
                                dtype=self.dtype, 
                                shape=(None,None,self.ydim), 
                                name='Yval_plc')
            # (val_task_batchsize, val_task_datasize, ydim)


    def make_variables(self):

        with self.graph.as_default():
            print("Use truncated normal initializer with stddev=0.01 for weights")
            self.approximator_param = tf.get_variable(
                initializer=tf.random.truncated_normal(
                    shape=[self.n_param], 
                    stddev=0.01, 
                    dtype=self.dtype),
                dtype=self.dtype, 
                name='approximator_param')


    def make_tensors(self):

        with self.graph.as_default():
                
            task = self.task

            Xtrain = tf.expand_dims(self.Xtrain_plc, axis=0)
            Ytrain = tf.expand_dims(self.Ytrain_plc, axis=0)
            # (1,ndataset,npoint,xdim)

            param = tf.expand_dims( tf.expand_dims(self.approximator_param, axis=0), axis=0 )
            # (1,1,n_param)

            updated_param = param

            # 1. compute k-step away from the predicted theta (either mean_theta or theta_samples)
            for i in range(self.n_step):
                # 1.1. train loss
                predicted_Ytrain = self.approximator.predict(Xtrain, updated_param)
                # (1,ndataset,npoint,ydim)

                training_loss_pertask = tf.reduce_mean( 
                        task.get_y_distance(predicted_Ytrain, Ytrain), 
                        axis=2)
                # (1, ndataset)

                if i == 0:
                    self.avg_train_loss = tf.reduce_mean(training_loss_pertask)

                # 1.2. gradient of train_loss
                train_loss_grad = []
                for j in range(self.training_task_batchsize):
                    train_loss_grad_j = tf.gradients(training_loss_pertask[0,j], updated_param)[0]
                    # (1,1,n_param)
                    
                    if i > 0:
                        train_loss_grad_j = tf.gather(train_loss_grad_j, indices=[j], axis=1)

                    train_loss_grad.append( train_loss_grad_j )

                train_loss_grad = tf.concat(train_loss_grad, axis=1)
                # (1,ndataset,n_param)

                updated_param = updated_param \
                    - train_loss_grad * self.stepsize

            if self.n_step == 0:
                # 1.1. train loss
                predicted_Ytrain = self.approximator.predict(Xtrain, updated_param)
                # (1,ndataset,npoint,ydim)

                self.avg_train_loss = tf.reduce_mean( 
                        task.get_y_distance(predicted_Ytrain, Ytrain) )


            self.updated_param = updated_param

            Xval = tf.expand_dims(self.Xval_plc, axis=0)
            Yval = tf.expand_dims(self.Yval_plc, axis=0)
            # (1,ndataset,npoint,xdim)

            # # compute the validation loss
            predicted_Yval = self.approximator.predict(Xval, self.updated_param)
            val_loss = tf.reduce_mean( 
                        task.get_y_distance(predicted_Yval, Yval), 
                        axis=2)
            # (1, ndataset)

            self.avg_val_loss = tf.reduce_mean(val_loss)

            self.train_vars = [self.approximator_param]

            self.train_op = tf.train.AdamOptimizer().minimize(
                                    self.avg_val_loss, 
                                    var_list=self.train_vars)


    def train(self, 
              n_train_iteration=1000, 
              path_to_load=None,
              path_to_save=None, 
              verbose=True):

        with self.graph.as_default():
                
            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())

                if path_to_load is not None:
                    with open(path_to_load, 'rb') as loadfile:
                        learned_params = pickle.load(loadfile)

                    self.approximator_param.load(learned_params["approximator_param"])
                    print("Load initializers from {}".format(path_to_load))                                    


                t = time.time()

                for i in range(n_train_iteration):

                    X_np, Y_np, _ \
                        = self.task.get_random_datasets(
                                self.training_task_batchsize, 
                                2 * self.training_task_datasize)

                    Xtrain_np = X_np[:,:self.training_task_datasize,:]
                    Ytrain_np = Y_np[:,:self.training_task_datasize,:]
                    Xval_np = X_np[:,self.training_task_datasize:,:]
                    Yval_np = Y_np[:,self.training_task_datasize:,:]

                    sess.run(self.train_op,
                        feed_dict = {
                            self.Xtrain_plc: Xtrain_np,
                            self.Ytrain_plc: Ytrain_np,
                            self.Xval_plc: Xval_np,
                            self.Yval_plc: Yval_np
                        })

                    avg_train_loss_np, avg_val_loss_np \
                        = sess.run([self.avg_train_loss, self.avg_val_loss],
                            feed_dict = {
                                self.Xtrain_plc: Xtrain_np,
                                self.Ytrain_plc: Ytrain_np,
                                self.Xval_plc: Xval_np,
                                self.Yval_plc: Yval_np
                            })

                    if np.isnan(avg_val_loss_np):
                        raise Exception("Loss is nan!")
                    
                    if verbose and i % 1000 == 0:
                        print("{:7d}. Loss: {:14.4f} {:14.4f} in {:14.4f}s".format(
                                            i, 
                                            avg_train_loss_np,
                                            avg_val_loss_np,
                                            time.time() - t))
                        sys.stdout.flush()

                        t = time.time()

                        if path_to_save is not None:
                            approximator_param_np = sess.run(self.approximator_param)
                            learned_params = {"approximator_param": approximator_param_np}

                            with open(path_to_save, 'wb') as savefile:
                                            
                                pickle.dump(learned_params, 
                                            savefile, 
                                            protocol=pickle.HIGHEST_PROTOCOL) 

                if path_to_save is not None:
                    approximator_param_np = sess.run(self.approximator_param)
                    learned_params = {"approximator_param": approximator_param_np}

                    with open(path_to_save, 'wb') as savefile:
                                    
                        pickle.dump(learned_params, 
                                    savefile, 
                                    protocol=pickle.HIGHEST_PROTOCOL) 
        return learned_params


    def predict_param(self, X, Y, learned_params):
        """
        X: (ntask, npoint, xdim)
        Y: (ntask, npoint, ydim)
        learned_params: dictionary 
            {"log_lscale", "log_sigma", "log_sigma0",
            "scalar_mean", "theta_u"}
        """
        with self.graph.as_default():
                
            with tf.Session() as sess:

                self.approximator_param.load(learned_params["approximator_param"])

                approximator_param_np, approximator_updated_param_np \
                    = sess.run([self.approximator_param, self.updated_param],
                            feed_dict = {
                                self.Xtrain_plc: X,
                                self.Ytrain_plc: Y
                            })

        return approximator_param_np, approximator_updated_param_np
    


