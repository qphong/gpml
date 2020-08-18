import os 

import numpy as np 
import tensorflow as tf 

import matplotlib.pyplot as plt 

import pickle 
import argparse

from gpml2 import GPML 
from maml import MAML

import cosine_task
import noisy_sine_task
import sensor_task
import cosine_line_task
import polynomial_task

from NetApproximator import NetApproximator 



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Meta-learning for regression problems.')

    parser.add_argument('--task', help='in {cosine, noisysine, cosineline, polynomial, sensorlight}',
                        required=False,
                        type=str,
                        default='cosine')
                       
    parser.add_argument('--method', help='in {gml, maml}',
                        required=False,
                        type=str,
                        default='gml')
                       
    parser.add_argument('--batchsize', help='training task batchsize, which is also validation task batchsize',
                        required=False,
                        type=int,
                        default=5)

    parser.add_argument('--datasize', help='training task datasize, which is also validation task datasize',
                        required=False,
                        type=int,
                        default=5)
                       
    parser.add_argument('--stepsize', help='stepsize for gradient ascent update (inner loop of maml)',
                        required=False,
                        type=float,
                        default=1e-3)

    parser.add_argument('--nstep', help='number of gradient ascent updates (inner loop of maml)',
                        required=False,
                        type=int,
                        default=1)

    parser.add_argument('--ntrain', help='number of training iteration',
                        required=False,
                        type=int,
                        default=100000)

    # only for gml
    parser.add_argument('--nu', help='method:gml - number of inducing tasks',
                        required=False,
                        type=int,
                        default=4)

    parser.add_argument('--sampling', help='method:gml - 1: use posterior samples, 0: use posterior mean for prediction',
                        required=False,
                        type=int,
                        default=0)

    parser.add_argument('--nparamsample', help='method:gml, sampling:1 - the number of posterior samples used for prediction',
                        required=False,
                        type=int,
                        default=5)
 
    parser.add_argument('--adaptivepivot', help='method:gml - 1: the pivot is selected adaptive based on training dataset, 0: the pivot is fixed',
                        required=False,
                        type=int,
                        default=1)

    parser.add_argument('--npivot', help='method:gml, adaptivepivot:0 - number of fixed pivots',
                        required=False,
                        type=int,
                        default=100)


    args = parser.parse_args()
    # print all arguments
    print('================================')
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print('================================')


    np.random.seed(0)
    tf.set_random_seed(1)

    dtype = tf.float64 


    if args.task == 'cosine':
        task = cosine_task 
    elif args.task == 'noisysine':
        task = noisy_sine_task
    elif  args.task == 'sensorlight':
        task = sensor_task.Sensor('light')
    elif args.task == 'cosineline':
        task = cosine_line_task
    elif args.task == 'polynomial':
        task = polynomial_task


    folder = "./{}".format(args.task)
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    xdim = task.xdim
    ydim = task.ydim

    print("For simplicity: assuming ydim = 1")
    assert ydim == 1, "only implemented for 1-d output"


    method = args.method

    training_task_batchsize = args.batchsize 
    training_task_datasize = args.datasize
    stepsize = args.stepsize
    n_step = args.nstep 
    ntrain = args.ntrain 

    if method == 'gml':
        n_inducing_tasks = args.nu 
        use_samples = args.sampling
        n_predicted_param_sample = args.nparamsample

        is_pivot_X_adaptive_in_Kuu = args.adaptivepivot
        npivot = args.npivot

        pivot_X, _, _ = task.get_random_datasets(ndataset=npivot, dataset_size=1)
        # (ndataset, dataset_size, xdim)
        pivot_X = pivot_X[:,0,:]
        # (npivot,xdim)


    if args.task == "noisysine":
        layer_sizes = [40, 40, 40, 1]
        activations = ['relu', 'relu', 'relu', 'linear']

    else:
        layer_sizes = [40, 40, 1]
        activations = ['relu', 'relu', 'linear']

    print("Model:")
    print("Layer size:  {}".format(layer_sizes))
    print("Activations: {}".format(activations))
    print("")
    
    netapprox = NetApproximator(xdim, 
                layer_sizes = layer_sizes, 
                activations = activations)


    if method == 'gml':
        metalearn = GPML(
            xdim = xdim,
            ydim = ydim,

            approximator = netapprox,
            task = task,

            n_inducing_tasks = n_inducing_tasks,

            training_task_batchsize = training_task_batchsize,
            training_task_datasize = training_task_datasize,

            stepsize = stepsize,
            n_step = n_step,

            use_samples = use_samples,
            n_predicted_param_sample = n_predicted_param_sample,

            is_pivot_X_adaptive_in_Kuu = is_pivot_X_adaptive_in_Kuu,
            pivot_X = pivot_X, # (nXu, xdim)
            
            dtype = dtype
        )

        folder = "{}/gml{}_{}_u{}_batch{}_k{}_nstep{}".format(folder, 
                "Bayes" if metalearn.use_samples else "",
                "adt" if metalearn.is_pivot_X_adaptive_in_Kuu else "fix", 
                metalearn.n_inducing_tasks,
                metalearn.training_task_batchsize,
                metalearn.training_task_datasize,
                metalearn.n_step
            )

        if not os.path.exists(folder):
            os.makedirs(folder)
            
        path_to_save = "{}/params.p".format(folder)
            
    elif method == 'maml':
        metalearn = MAML(
            xdim = task.xdim,
            ydim = task.ydim,

            approximator = netapprox,
            task = task,

            training_task_batchsize = training_task_batchsize,
            training_task_datasize = training_task_datasize,

            stepsize = stepsize,
            n_step = n_step,

            dtype = dtype
        )

        folder = "{}/maml_batch{}_k{}_nstep{}".format(folder, 
                    metalearn.training_task_batchsize,
                    metalearn.training_task_datasize,
                    metalearn.n_step)
        if not os.path.exists(folder):
            os.makedirs(folder)

        path_to_save = "{}/params.p".format(folder)


    path_to_load = None
    
    learned_params = metalearn.train(n_train_iteration=ntrain, 
                        path_to_save=path_to_save,
                        path_to_load=path_to_load)


    # get the testing performance after training
    n_test_tasks = metalearn.training_task_batchsize
    n_test_datapoints = 1000

    if args.task.startswith("sensor"):
        # metalearn.training_task_datasize + n_test_datapoints
        #   is ignored in this case, as we use all test task's dataset
        X_np, Y_np, test_params_np = task.get_random_datasets(
            n_test_tasks,
            metalearn.training_task_datasize + n_test_datapoints,
            is_training=False
        )
    else:
        X_np, Y_np, test_params_np = task.get_random_datasets(
            n_test_tasks,
            metalearn.training_task_datasize + n_test_datapoints
        )

    X_train_np = X_np[:,:metalearn.training_task_datasize,:]
    Y_train_np = Y_np[:,:metalearn.training_task_datasize,:]
    X_test_np = X_np[:,metalearn.training_task_datasize:,:]
    Y_test_np = Y_np[:,metalearn.training_task_datasize:,:]

    if method == 'gml':
        mean_theta_np, std_theta_np, theta_samples_np, updated_theta_samples_np \
            = metalearn.predict_param(
                X_train_np, 
                Y_train_np, 
                learned_params)

        theta_np = theta_samples_np
        updated_theta_np = updated_theta_samples_np
    else:        
        approximator_param_np, approximator_updated_param_np \
            = metalearn.predict_param(
                X_train_np, 
                Y_train_np, 
                learned_params)
        theta_np = approximator_param_np.reshape(1,1,approximator_param_np.shape[-1])
        updated_theta_np = approximator_updated_param_np


    graph = tf.Graph()

    with graph.as_default():
        x_plc = tf.placeholder(dtype=dtype,
                    shape=(None, None, None, task.xdim))
        # (n_theta_sample|1, ndataset, npoint, xdim)

        param_plc = tf.placeholder(dtype=dtype,
                    shape=(None, None, netapprox.n_param))    
        # (n_theta_sample, ndataset|1, n_param)

        predicted_y = netapprox.predict(x_plc, param_plc)

        with tf.Session() as sess:

            predicted_y_np = sess.run(predicted_y,
                                feed_dict = {
                                    x_plc: np.expand_dims(
                                            X_test_np, axis=0) ,
                                    param_plc: theta_np
                                })

            predicted_updated_y_np = sess.run(predicted_y,
                                feed_dict = {
                                    x_plc: np.expand_dims(
                                            X_test_np, axis=0) ,
                                    param_plc: updated_theta_np
                                })

        print("Average MSE:", np.mean( np.square( predicted_y_np - Y_test_np ) ))
        print("            ", np.mean( np.square( predicted_updated_y_np - Y_test_np ) ))


