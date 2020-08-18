import os 

import numpy as np 
import tensorflow as tf 

import pickle 

from gpml2 import GPML 
from maml import MAML 
from tqdm import tqdm

import cosine_task
import cosine_line_task
import polynomial_task 
import sensor_task 
import noisy_sine_task

from NetApproximator import NetApproximator 


method = "gml"
task_name = "polynomial"

if task_name == "cosine":
    task = cosine_task
elif task_name == "noisysine":
    task = noisy_sine_task
elif task_name == "cosineline":
    task = cosine_line_task
elif task_name == "polynomial":
    task = polynomial_task 
elif task_name == 'sensorlight':
    task = sensor_task.Sensor('light', selected_hour='all')


np.random.seed(0)
tf.set_random_seed(1)


dtype = tf.float64 
xdim = task.xdim
ydim = task.ydim


n_inducing_tasks = 8
training_task_batchsize = 5
training_task_datasize = 5
n_step = 1
use_samples = True
is_pivot_X_adaptive_in_Kuu = True
n_pivot_X = 100 # used to create pivot_X if pivot_X is not adaptive


if method == "gpml" or method == "gml":

    folder = "{}/gml{}_{}_u{}_batch{}_k{}_nstep{}".format(
        task_name,
        "Bayes" if use_samples else "",
        "adt" if is_pivot_X_adaptive_in_Kuu else "fix",
        n_inducing_tasks,
        training_task_batchsize,
        training_task_datasize,
        n_step
    )

elif method == "maml":

    folder = "{}/maml_batch{}_k{}_nstep{}".format(task_name, 
                    training_task_batchsize,
                    training_task_datasize,
                    n_step)

path_to_load = "{}/params.p".format(folder)

with open(path_to_load, "rb") as readfile:
    print("Load params from {}".format(path_to_load))
    learned_params = pickle.load(readfile)


print("For simplicity: assuming ydim = 1")
assert ydim == 1, "only handle 1d output"

dtype = tf.float64

if task_name == "noisysine":
    netapprox = NetApproximator(xdim, 
                layer_sizes = [40, 40, 40, 1], 
                activations = ['relu', 'relu', 'relu', 'linear'])
else:
    netapprox = NetApproximator(xdim, 
                layer_sizes = [40, 40, 1], 
                activations = ['relu', 'relu', 'linear'])

if method == "gpml" or method == "gml":

    pivot_X, _, _ = task.get_random_datasets(1, n_pivot_X)
    pivot_X = pivot_X[0,...]

    metalearn = GPML(
        xdim = task.xdim,
        ydim = task.ydim,

        approximator = netapprox,
        task = task,

        n_inducing_tasks = n_inducing_tasks,

        training_task_batchsize = training_task_batchsize,
        training_task_datasize = training_task_datasize,

        stepsize = 1e-3,
        n_step = n_step,

        use_samples = use_samples,
        n_predicted_param_sample = 10,

        is_pivot_X_adaptive_in_Kuu = is_pivot_X_adaptive_in_Kuu,
        pivot_X = pivot_X, # (nXu, xdim)
        
        dtype = dtype
    )

elif method == "maml":

    metalearn = MAML(
        xdim = task.xdim,
        ydim = task.ydim,

        approximator = netapprox,
        task = task,

        training_task_batchsize = training_task_batchsize,
        training_task_datasize = training_task_datasize,

        stepsize = 1e-3,
        n_step = n_step,

        dtype = dtype
    )

    n_predicted_param_sample = 1

    
# Testing with new tasks

graph = tf.Graph()

with graph.as_default():
    x_plc = tf.placeholder(dtype=dtype,
                shape=(None, None, None, task.xdim))
    # (n_theta_sample|1, ndataset, npoint, xdim)

    param_plc = tf.placeholder(dtype=dtype,
                shape=(None, None, netapprox.n_param))    
    # (n_theta_sample, ndataset|1, n_param)

    predicted_y = netapprox.predict(x_plc, param_plc)
    # (n_theta_sample, ndataset, npoint, ydim)



# Evaluated on a large number of tesk tasks
seed = 0
np.random.seed(seed)
if task_name.startswith("sensor"):
    n_total_tesk_task = 1000
else:
    n_total_tesk_task = 100


n_test_tasks = metalearn.training_task_batchsize
n_repetitions = int(n_total_tesk_task / n_test_tasks)
n_total_tesk_task = n_repetitions * n_test_tasks

print("n_repetitions: ", n_repetitions)
print("n_test_task: ", n_test_tasks)
print("n_total_test_task", n_total_tesk_task)

n_test_datapoints = 1000

path_mse = "{}/mse_ntask{}_npoint{}_seed{}.p".format(
                folder, 
                n_total_tesk_task, 
                n_test_datapoints, 
                seed)


mse_all = np.zeros(n_total_tesk_task)
updated_mse_all = np.zeros(n_total_tesk_task)

for rep_idx in tqdm(range(n_repetitions)):

    if task_name.startswith("sensor"):
        X_np, Y_np, test_params_np = task.get_random_datasets(
            n_test_tasks,
            metalearn.training_task_datasize + n_test_datapoints,
            is_training=False,
        )

        n_test_tasks = X_np.shape[0]
        n_test_datapoints = X_np.shape[1] - metalearn.training_task_datasize

    else:
        X_np, Y_np, test_params_np = task.get_random_datasets(
            n_test_tasks,
            metalearn.training_task_datasize + n_test_datapoints
        )

    X_train_np = X_np[:,:metalearn.training_task_datasize,:]
    Y_train_np = Y_np[:,:metalearn.training_task_datasize,:]
    X_test_np = X_np[:,metalearn.training_task_datasize:,:]
    Y_test_np = Y_np[:,metalearn.training_task_datasize:,:]


    if method == "gpml" or method == "gml":
        mean_theta_np, std_theta_np, theta_samples_np, updated_theta_samples_np \
            = metalearn.predict_param(
                X_train_np, 
                Y_train_np, 
                learned_params)
        # theta_samples_np: (n_theta_sample, ndataset, ntheta)

    elif method == "maml":
        theta_samples_np, updated_theta_samples_np \
            = metalearn.predict_param(
                X_train_np, 
                Y_train_np, 
                learned_params)

        theta_samples_np = theta_samples_np.reshape(1,1,-1)

    with tf.Session(graph=graph) as sess:
        predicted_y_np = sess.run(predicted_y,
                            feed_dict = {
                                x_plc: np.expand_dims(
                                        X_test_np, axis=0) ,
                                param_plc: theta_samples_np
                            })
        # (n_theta_sample, ndataset, npoint, ydim)

        predicted_updated_y_np = sess.run(predicted_y,
                            feed_dict = {
                                x_plc: np.expand_dims(
                                        X_test_np, axis=0) ,
                                param_plc: updated_theta_samples_np
                            })
        # (n_theta_sample, ndataset, npoint, ydim)

        # Y_test_np.shape = (ndataset, npoint, ydim)

        mse = np.mean(
            np.mean(
                np.mean(np.square(predicted_y_np - Y_test_np), axis=-1), # avg ydim
                axis=-1), # avg npoint
            axis = 0) # avg theta_sample
        # (ndataset,)
        
        updated_mse = np.mean(
            np.mean(
                np.mean(np.square(predicted_updated_y_np - Y_test_np), axis=-1),
                axis=-1),
            axis = 0)
        # (ndataset,)

        mse_all[(rep_idx*n_test_tasks):(rep_idx*n_test_tasks + n_test_tasks)] = mse
        updated_mse_all[(rep_idx*n_test_tasks):(rep_idx*n_test_tasks + n_test_tasks)] = updated_mse

print("MSE.shape = ", mse_all.shape)
print("MSE:         mean {} std {}".format(np.mean(mse_all), np.std(mse_all)))
print("MSE updated: mean {} std {}".format(np.mean(updated_mse_all), np.std(updated_mse_all)))


with open(path_mse, "wb") as writefile:
    pickle.dump({"mse": mse_all,
                 "updated_mse": updated_mse_all}, 
                writefile, 
                protocol=pickle.HIGHEST_PROTOCOL)
