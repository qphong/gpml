import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp
 
import time 
import pickle
import sys

import util 
import gp_util


class GPML():

    def __init__(self, 
            xdim, ydim,

            approximator, # the machine learning model
            task, # task to learn

            n_inducing_tasks, # nu
            
            training_task_batchsize, # ndataset
            training_task_datasize, # npoint
            
            stepsize, # opt_step_size
            n_step, # n_stepsize

            use_samples = False,
            n_predicted_param_sample = 100, # n_theta_sample

            is_pivot_X_adaptive_in_Kuu = True,
            pivot_X = None, # (nXu, xdim)

            dtype = tf.float32):

        self.xdim = xdim 
        self.ydim = ydim

        self.n_inducing_tasks = n_inducing_tasks

        self.training_task_batchsize = training_task_batchsize
        self.training_task_datasize = training_task_datasize

        self.stepsize = stepsize
        self.n_step = n_step 
        
        self.use_samples = use_samples
        self.n_predicted_param_sample = n_predicted_param_sample
        
        self.is_pivot_X_adaptive_in_Kuu = is_pivot_X_adaptive_in_Kuu
        """
        if is_pivot_X_adaptive_in_Kuu:
            set pivot_X for theta_u to be the same as training inputs of the training task
        else:
            set a common set of pivot_X for all theta_u, which is independent from training tasks
            need input pivot_X != None
        """
        self.pivot_X_toload = pivot_X


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
    
            self.log_lscale = tf.get_variable(
                            dtype=self.dtype, 
                            shape=(self.n_param,), 
                            name='lengthscale')
            self.log_sigma = tf.get_variable(
                            dtype=self.dtype, 
                            shape=(self.n_param,), 
                            name='signal_variance')
            self.log_sigma0 = tf.get_variable(
                            dtype=self.dtype, 
                            shape=(self.n_param,), 
                            name='noise_variance')

            print("Use truncated normal initializer with stddev=0.01 for scalar_mean")
            self.scalar_mean = tf.get_variable(
                            initializer=tf.random.truncated_normal(
                                shape=[self.n_param], 
                                stddev=0.01, 
                                dtype=self.dtype),
                            dtype=self.dtype, 
                            name='scalar_mean')


            self.lscale = tf.exp(self.log_lscale)
            self.sigma = tf.exp(self.log_sigma) + 1e-6
            self.sigma0 = tf.exp(self.log_sigma0) + 1e-4

            print("Use truncated normal initializer with stddev=0.01 for theta_u")
            self.theta_u = tf.get_variable(
                            initializer=tf.random.truncated_normal(
                                shape=[self.n_inducing_tasks, self.n_param], 
                                stddev=0.01, 
                                dtype=self.dtype),
                            dtype=self.dtype, 
                            name='theta_u')


    def make_tensors(self):

        if self.is_pivot_X_adaptive_in_Kuu:
            # inducing input (different from the inducing tasks)
            self.pivot_X = self.Xtrain_plc # Xu
        elif self.pivot_X_toload is None:
            raise Exception("Require pivot_X if is_pivot_X_adaptive_in_Kuu == True")

        with self.graph.as_default():

            if self.is_pivot_X_adaptive_in_Kuu:
                self.invKu = None

            else:
                # pivot_X is the same for all training datasets/tasks
                # self.pivot_X (nXu,xdim)
                self.pivot_X_tobeloaded = tf.get_variable(
                                            dtype=self.dtype, 
                                            shape=(self.pivot_X_toload.shape[0], self.xdim), 
                                            name="pivot_X_tobeloaded")
                self.pivot_X = tf.expand_dims(self.pivot_X_tobeloaded, axis=0)
                # (1, nXu, xdim)

                dist_u = gp_util.compute_dmm(
                        tf.expand_dims(self.theta_u, axis=1), # (n_inducing_tasks, 1, n_param)
                        tf.expand_dims(self.pivot_X, axis=0), # (1, 1, nXu, xdim)
                        self.approximator.predict, 
                        self.task.get_y_distance)
                # 1,n_inducing_tasks,n_inducing_tasks,npoint

                Ku = gp_util.compute_K(dist_u, 
                                    self.lscale, 
                                    self.sigma)
                # 1,n_inducing_tasks,n_inducing_tasks,n_param

                Ku = tf.transpose(Ku, perm=[3,0,1,2])
                # n_param,1,n_inducing_tasks,n_inducing_tasks

                # add noise
                noise = tf.eye(self.n_inducing_tasks, 
                            batch_shape=(self.n_param, 1), 
                            dtype=self.dtype) \
                        * tf.reshape(self.sigma0, shape=(self.n_param,1,1,1))
                Ku = Ku + noise

                invKu = util.multichol2inv(
                    tf.reshape(Ku, 
                        shape=(self.n_param, 
                                self.n_inducing_tasks,
                                self.n_inducing_tasks)),
                    self.n_param, 
                    dtype=self.dtype)
                # ntheta,nu,nu

                invKu = tf.expand_dims(invKu, axis=1)
                # ntheta,1,nu,nu

                self.invKu = tf.tile(invKu, multiples=(1,self.training_task_batchsize,1,1))
                # (ntheta, ndataset, nu,nu)

        self.mean_theta = [None] * self.training_task_batchsize
        self.std_theta = [None] * self.training_task_batchsize
        
        self.theta_samples = [None] * self.training_task_batchsize
        self.updated_theta_samples = [None] * self.training_task_batchsize
        
        self.train_losses = [None] * self.training_task_batchsize
        self.val_losses = [None] * self.training_task_batchsize

        for dataset_idx in range(self.training_task_batchsize):
            self.compute_loss_per_dataset(dataset_idx)

        with self.graph.as_default():

            self.mean_theta = tf.concat(self.mean_theta, axis=0)
            self.std_theta = tf.concat(self.std_theta, axis=0)
            # (ndataset, ntheta)

            self.theta_samples = tf.concat(self.theta_samples, axis=1)
            self.updated_theta_samples = tf.concat(self.updated_theta_samples, axis=1)
            # (n_theta_samples, ndataset, ntheta)

            self.avg_train_loss = tf.reduce_mean(self.train_losses)
            self.avg_val_loss = tf.reduce_mean(self.val_losses)
            # (ndataset)

        with self.graph.as_default():
            
            self.train_vars = [self.theta_u, 
                        self.scalar_mean, 
                        self.log_lscale, 
                        self.log_sigma, 
                        self.log_sigma0]

            self.train_op = tf.train.AdamOptimizer(1e-3).minimize(
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

                # load log_lscale, log_sigma, log_sigma0
                log_lscale_np = np.ones(self.n_param) * np.log(1.0)
                log_sigma_np = np.ones(self.n_param) * np.log(1e-2)
                log_sigma0_np = np.ones(self.n_param) * np.log(1.0)

                self.log_lscale.load(log_lscale_np, sess)
                self.log_sigma.load(log_sigma_np, sess)
                self.log_sigma0.load(log_sigma0_np, sess)

                if not self.is_pivot_X_adaptive_in_Kuu:
                    self.pivot_X_tobeloaded.load(self.pivot_X_toload, sess)

                if path_to_load is not None:
                    with open(path_to_load, 'rb') as loadfile:
                        learned_params = pickle.load(loadfile)

                    self.log_lscale.load(learned_params["log_lscale"], 
                                        sess)
                    self.log_sigma.load(learned_params["log_sigma"], 
                                        sess)
                    self.log_sigma0.load(learned_params["log_sigma0"],
                                        sess)
                    self.scalar_mean.load(learned_params["scalar_mean"],
                                        sess)
                    self.theta_u.load(learned_params["theta_u"],
                                    sess)

                    if not learned_params["is_pivot_X_adaptive_in_Kuu"]:    
                        self.pivot_X_tobeloaded.load(learned_params["pivot_X"])

                    print("Load initializers from {}".format(path_to_load))                                    

                t = time.time()

                for i in range(n_train_iteration):

                    X_np, Y_np, info \
                        = self.task.get_random_datasets(
                                self.training_task_batchsize, 
                                2 * self.training_task_datasize)

                    # print("{}. {}".format(i, info.keys()))

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

                    avg_train_loss_np, avg_val_loss_np = sess.run([self.avg_train_loss, self.avg_val_loss],
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

                        log_lscale_np, log_sigma_np, log_sigma0_np, \
                        scalar_mean_np, theta_u_np \
                                    = sess.run([self.log_lscale, 
                                                self.log_sigma, 
                                                self.log_sigma0, 
                                                self.scalar_mean, 
                                                self.theta_u])

                        if not self.is_pivot_X_adaptive_in_Kuu:
                            pivot_X_np = sess.run(self.pivot_X_tobeloaded)
                        else:
                            pivot_X_np = None

                        learned_params = {"log_lscale": log_lscale_np,
                                        "log_sigma": log_sigma_np,
                                        "log_sigma0": log_sigma0_np,
                                        "scalar_mean": scalar_mean_np,
                                        "is_pivot_X_adaptive_in_Kuu": self.is_pivot_X_adaptive_in_Kuu,
                                        "pivot_X": pivot_X_np,
                                        "theta_u": theta_u_np}

                        if path_to_save is not None:
                            with open(path_to_save, 'wb') as savefile:
                                            
                                pickle.dump(learned_params, 
                                            savefile, 
                                            protocol=pickle.HIGHEST_PROTOCOL) 

                log_lscale_np, log_sigma_np, log_sigma0_np, \
                scalar_mean_np, theta_u_np \
                            = sess.run([self.log_lscale, 
                                        self.log_sigma, 
                                        self.log_sigma0, 
                                        self.scalar_mean, 
                                        self.theta_u])

                if not self.is_pivot_X_adaptive_in_Kuu:
                    pivot_X_np = sess.run(self.pivot_X_tobeloaded)
                else:
                    pivot_X_np = None
                    
                learned_params = {"log_lscale": log_lscale_np,
                                "log_sigma": log_sigma_np,
                                "log_sigma0": log_sigma0_np,
                                "scalar_mean": scalar_mean_np,
                                "is_pivot_X_adaptive_in_Kuu": self.is_pivot_X_adaptive_in_Kuu,
                                "pivot_X": pivot_X_np,
                                "theta_u": theta_u_np}

                if path_to_save is not None:
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

                self.log_lscale.load(learned_params["log_lscale"], 
                                    sess)
                self.log_sigma.load(learned_params["log_sigma"], 
                                    sess)
                self.log_sigma0.load(learned_params["log_sigma0"],
                                    sess)
                self.scalar_mean.load(learned_params["scalar_mean"],
                                    sess)
                self.theta_u.load(learned_params["theta_u"],
                                sess)

                if not learned_params["is_pivot_X_adaptive_in_Kuu"]:    
                    self.pivot_X_tobeloaded.load(learned_params["pivot_X"])
                    
                mean_theta_np, std_theta_np, \
                theta_samples_np, \
                updated_theta_samples_np \
                    = sess.run([self.mean_theta, 
                            self.std_theta, 
                            self.theta_samples, 
                            self.updated_theta_samples], 
                            feed_dict = {
                                self.Xtrain_plc: X,
                                self.Ytrain_plc: Y
                            })

        return mean_theta_np, std_theta_np, theta_samples_np, updated_theta_samples_np



    def compute_loss_per_dataset(self, dataset_idx):
        """
        create self.pivot_X before calling this function
        of shape (ndataset, npoint, xdim)
        """
        with self.graph.as_default():
            
            if self.invKu is None:
                assert self.is_pivot_X_adaptive_in_Kuu
                
                pivot_X_i = tf.gather(self.pivot_X, indices=[dataset_idx], axis=0)
                # (1, npoint, xdim)

                dist_u = gp_util.compute_dmm(
                        tf.expand_dims(self.theta_u, axis=1), # (n_inducing_tasks, 1, n_param)
                        tf.expand_dims(pivot_X_i, axis=0), # (1, 1, npoint, xdim)
                        self.approximator.predict, 
                        self.task.get_y_distance)
                # (1, n_inducing_tasks, n_inducing_tasks, npoint)

                Ku = gp_util.compute_K(dist_u, self.lscale, self.sigma)
                # 1, nu, nu, n_param

                Ku = tf.transpose(Ku, perm=[3,0,1,2])
                # n_param,1,nu,nu
            
                # add noise
                noise = tf.eye(self.n_inducing_tasks, 
                            batch_shape=(self.n_param, 1), 
                            dtype=self.dtype) \
                        * tf.reshape(self.sigma0, shape=(self.n_param,1,1,1))
                Ku = Ku + noise


                invKu_i = util.multichol2inv(
                    tf.reshape(Ku, 
                        shape=(self.n_param, 
                                self.n_inducing_tasks,
                                self.n_inducing_tasks)),
                    self.n_param, 
                    dtype=self.dtype)
                # n_param,nu,nu
                
                invKu_i =tf.reshape(invKu_i, 
                            shape=(self.n_param,
                                1,
                                self.n_inducing_tasks,
                                self.n_inducing_tasks))
                # (ntheta, 1, nu,nu)
            
            else:
                invKu_i = tf.gather(self.invKu, 
                                indices=[dataset_idx], 
                                axis=1)

            Xtrain_i_plc = tf.gather(self.Xtrain_plc, indices=[dataset_idx], axis=0)
            Ytrain_i_plc = tf.gather(self.Ytrain_plc, indices=[dataset_idx], axis=0)

            mean_theta, var_theta = gp_util.compute_mean_var_theta(
                                        Xtrain_i_plc, 
                                        Ytrain_i_plc,
                                        self.theta_u,
                                        self.scalar_mean, 
                                        self.approximator.predict,
                                        self.task.get_y_distance,
                                        self.lscale, 
                                        self.sigma,
                                        invKu_i
                                    )
            # (ntheta,1)

            if self.n_step == 0:
                var_theta = var_theta + tf.reshape(self.sigma0, tf.shape(var_theta))

            mean_theta = tf.transpose(mean_theta)
            # (1,ntheta)
            mean_theta = mean_theta + self.scalar_mean

            var_theta = tf.transpose(var_theta)
            std_theta = tf.sqrt(var_theta)
            # (1,ntheta)

            self.mean_theta[dataset_idx] = mean_theta 
            self.std_theta[dataset_idx] = std_theta 

            if self.use_samples:
                print("Use GP posterior samples to train")
                theta_samples = tfp.distributions.Normal(
                                    loc=mean_theta, 
                                    scale=std_theta).sample(self.n_predicted_param_sample)
            else:
                print("Use mean of GP to train")
                self.n_predicted_param_sample = 1
                theta_samples = tf.expand_dims(mean_theta, axis=0)
            # (n_predicted_param_sample,1,ntheta)

            Xtrain = tf.expand_dims(Xtrain_i_plc, axis=0)
            Ytrain = tf.expand_dims(Ytrain_i_plc, axis=0)
            # (1,1,npoint,xdim)

            self.theta_samples[dataset_idx] = theta_samples

            updated_theta_samples = theta_samples
            # (n_theta_sample,1,ntheta)

            for i in range(self.n_step):
                # 1. compute k-step away from the predicted theta (either mean_theta or theta_samples)

                # 1.1. train loss
                predicted_Ytrain = self.approximator.predict(Xtrain, updated_theta_samples)
                # (n_theta_sample,1,npoint,ydim)

                training_loss_pertask = tf.reduce_mean( self.task.get_y_distance(predicted_Ytrain, Ytrain), axis=2)
                # (n_theta_sample, 1)

                if i == 0:
                    self.train_losses[dataset_idx] = tf.reduce_mean(training_loss_pertask)

                # 1.2. gradient of train_loss
                train_loss_grad = tf.gradients(
                                    training_loss_pertask, 
                                    updated_theta_samples)[0]
                # (n_theta_sample, 1, ntheta)

                updated_theta_samples = updated_theta_samples \
                    - train_loss_grad * self.stepsize

            if self.n_step == 0:
                # 1.1. train loss
                predicted_Ytrain = self.approximator.predict(Xtrain, updated_theta_samples)
                # (n_theta_sample,1,npoint,ydim)

                self.train_losses[dataset_idx] = tf.reduce_mean( 
                    self.task.get_y_distance(predicted_Ytrain, Ytrain) )


            self.updated_theta_samples[dataset_idx] = updated_theta_samples
            # (n_theta_sample,1,ntheta)

            Xval_i = tf.gather(self.Xval_plc, indices=[dataset_idx], axis=0)
            Yval_i = tf.gather(self.Yval_plc, indices=[dataset_idx], axis=0)
            
            Xval_i = tf.expand_dims(Xval_i, axis=0)
            Yval_i = tf.expand_dims(Yval_i, axis=0)
            # (1,1,npoint,xdim)

            # # compute the validation loss
            predicted_Yval = self.approximator.predict(Xval_i, self.updated_theta_samples[dataset_idx])
            val_loss = tf.reduce_mean( 
                        self.task.get_y_distance(predicted_Yval, Yval_i), 
                        axis=2)
            # (n_theta_sample, 1)

            self.val_losses[dataset_idx] = tf.reduce_mean(val_loss)
