import os 


import numpy as np 
import tensorflow as tf 

import pickle

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

plt.style.use('seaborn')

import seaborn as sns
colors = sns.color_palette()


import pickle 

from gpml2 import GPML 

import cosine_task
import cosine_line_task
import polynomial_task 
import sensor_task 
import noisy_sine_task 


from NetApproximator import NetApproximator 


"""
the reason we cannot plot the covariance matrix is because covariance matrices are different
    for different parameters
therefore, we plot the mse between tasks instead
"""

task_name = "cosine"
savefig = True

if task_name == "cosine":
    task = cosine_task
elif task_name == "cosineline":
    task = cosine_line_task
elif task_name == "noisysine":
    task = noisy_sine_task
elif task_name == "polynomial":
    task = polynomial_task 
elif task_name == 'sensorlight':
    task = sensor_task.Sensor('light', selected_hour='all')

np.random.seed(0)
tf.set_random_seed(1)


dtype = tf.float64 
xdim = task.xdim
ydim = task.ydim


n_inducing_tasks = 2
training_task_batchsize = 5
training_task_datasize = 5
n_step = 1
use_samples = True
is_pivot_X_adaptive_in_Kuu = True
n_pivot_X = 100 # used to create pivot_X if pivot_X is not adaptive


folder = "{}/gml{}_{}_u{}_batch{}_k{}_nstep{}".format(
    task_name,
    "Bayes" if use_samples else "",
    "adt" if is_pivot_X_adaptive_in_Kuu else "fix",
    n_inducing_tasks,
    training_task_batchsize,
    training_task_datasize,
    n_step
)


path_to_load = "{}/params.p".format(folder)

with open(path_to_load, "rb") as readfile:
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

pivot_X, _, _ = task.get_random_datasets(1, n_pivot_X)
pivot_X = pivot_X[0,...]


# Testing with new tasks

graph = tf.Graph()

with graph.as_default():
    x_plc = tf.placeholder(dtype=dtype,
                shape=(None, None, None, task.xdim))
    # (n_theta_sample|1, ndataset, npoint, xdim)

    param_plc = tf.placeholder(dtype=dtype,
                shape=(None, None, netapprox.n_param))    
    # (n_theta_sample, ndataset|1, n_param)

    predicted_y = netapprox.predict(
            x_plc, 
            param_plc,
            )
    # (n_theta_sample, ndataset, npoint, ydim)


# Evaluated on a large number of tesk tasks
seed = 0
np.random.seed(seed)
n_total_tesk_task = 500

n_test_tasks = training_task_batchsize
n_repetitions = int(n_total_tesk_task / n_test_tasks)
n_total_tesk_task = n_repetitions * n_test_tasks

print("n_repetitions: ", n_repetitions)
print("n_test_task: ", n_test_tasks)
print("n_total_test_task", n_total_tesk_task)

n_test_datapoints = training_task_datasize

path_mse = "{}/mse_ntask{}_npoint{}_seed{}.p".format(
                folder, 
                n_total_tesk_task, 
                n_test_datapoints, 
                seed)


mse_tasks_all = []
mse_2u_all = np.zeros([n_inducing_tasks, n_total_tesk_task])
mse_2mean_all = np.zeros(n_total_tesk_task)
mse_u2mean_all = np.zeros([n_inducing_tasks, n_total_tesk_task])
mse_u2u_all = np.zeros([n_inducing_tasks, n_inducing_tasks-1, n_total_tesk_task])

for rep_idx in range(n_repetitions):
        
    X_np, Y_np, test_params_np = task.get_random_datasets(
        n_test_tasks,
        training_task_datasize + n_test_datapoints
    )
    # X_np: n_test_tasks, datasize, xdim
    # Y_np: n_test_tasks, datasize, ydim

    X_train_np = X_np[:,:training_task_datasize,:]
    Y_train_np = Y_np[:,:training_task_datasize,:]

    # compute distance between Y_train of training tasks
    # Y_train_np: n_dataset, 
    Y_train_np1 = np.expand_dims(Y_train_np, axis=1)
    Y_train_np1 = np.tile(Y_train_np1, reps=(1, n_test_tasks, 1, 1))

    mse_tasks = np.mean(
            np.mean(np.square(Y_train_np1 - Y_train_np), axis=-1), # avg ydim
            axis=-1) # avg npoint
    # (ndataset,ndataset)
    mse_tasks = mse_tasks.reshape(-1,)
    # (ndataset*2)
    mse_tasks_all.append(mse_tasks)


    # compute distance beween Y_train of training tasks and their predicted y 
    # from theta_u and scalar_mean
    with tf.Session(graph=graph) as sess:
        predicted_y_np = sess.run(predicted_y,
                            feed_dict = {
                                x_plc: np.expand_dims(
                                        X_train_np, axis=0),
                                param_plc: learned_params['theta_u'].reshape(n_inducing_tasks,1,netapprox.n_param)
                            })
        # (n_theta_u, ndataset, npoint, ydim)


        mean_y_np = sess.run(predicted_y,
                            feed_dict = {
                                x_plc: np.expand_dims(
                                        X_train_np, axis=0) ,
                                param_plc: learned_params['scalar_mean'].reshape(1,1,netapprox.n_param)
                            })


        mse_2u = np.mean(
                np.mean(np.square(predicted_y_np - Y_train_np), axis=-1), # avg ydim
                axis=-1) # avg npoint
        # (n_theta_u,ndataset)
        
        mse_2mean = np.mean(
                np.mean(np.square(mean_y_np - Y_train_np), axis=-1), # avg ydim
                axis=-1) # avg npoint
        # (1,ndataset)

        mse_u2mean = np.mean(
                np.mean(np.square(predicted_y_np - mean_y_np), axis=-1), # avg ydim
                axis=-1) # avg npoint
        # (n_theta_u, ndataset)

        for i in range(n_inducing_tasks):
            idx_not_i = list(range(i)) + list(range(i+1,n_inducing_tasks))
            mse_u2u = np.mean(
                    np.mean(np.square(predicted_y_np[idx_not_i,...] - predicted_y_np[i,...]), axis=-1), # avg ydim
                    axis=-1) # avg npoint
            # (n_theta_u-1, ndataset)
        
            mse_u2u_all[i,:,(rep_idx*n_test_tasks):(rep_idx*n_test_tasks + n_test_tasks)] \
                = mse_u2u

        mse_2u_all[:,(rep_idx*n_test_tasks):(rep_idx*n_test_tasks + n_test_tasks)] \
            = mse_2u
        mse_2mean_all[(rep_idx*n_test_tasks):(rep_idx*n_test_tasks + n_test_tasks)] \
            = mse_2mean[0,:]
        mse_u2mean_all[:,(rep_idx*n_test_tasks):(rep_idx*n_test_tasks + n_test_tasks)] \
            = mse_u2mean

mse_tasks_all = np.concatenate(mse_tasks_all, axis=0)

"""
mse_2u_all: (ntheta,ndataset)
mse_2mean_all: (ndataset)
mse_u2mean_all: (ntheta,ndataset)
mse_tasks_all: (n_dataset_pair)
"""

prefix = "{}{}_u{}_step{}".format(
    task_name, 
    "Bayes" if use_samples else "",
    n_inducing_tasks,
    n_step)

# plot the mse between inducing tasks and the training tasks
fig, ax = plt.subplots(figsize=(2.5,2), tight_layout=True)

sns.kdeplot(mse_tasks_all, shade=True, ax=ax)
# sns.kdeplot(mse_2mean_all, ax=ax, label='mean')

for i in range(n_inducing_tasks):
    sns.kdeplot(mse_2u_all[i,:], ax=ax, label=r"t$\rm _{}$".format(i), legend=True)

# ax.legend()

if savefig:
    fig.savefig("img/{}_dist_induce_vs_traintask.pdf".format(prefix))


# plot the mse between inducing tasks and the training tasks
fig, ax = plt.subplots(figsize=(2.5,2), tight_layout=True)

sns.kdeplot(mse_tasks_all, shade=True, ax=ax)
ax.get_lines()[0].remove()
sns.kdeplot(np.min(mse_2u_all, axis=0), ax=ax)


pickle.dump(
    {
        "min_dist": np.min(mse_2u_all, axis=0),
        "mse_tasks_all": mse_tasks_all
    },
    open('{}_mindist_induce_vs_traintask.pkl'.format(prefix), 'wb'))


if savefig:
    fig.savefig("img/{}_mindist_induce_vs_traintask.pdf".format(prefix))



# plot the mse between inducing tasks and the mean
fig, ax = plt.subplots(figsize=(2.5,2), tight_layout=True)

sns.kdeplot(mse_tasks_all, shade=True, ax=ax)
ax.get_lines()[0].remove()

for i in range(n_inducing_tasks):
    sns.kdeplot(mse_u2mean_all[i,:], ax=ax, label=r"t$\rm _{}$".format(i), legend=True)

if savefig:
    fig.savefig("img/{}_dist_induce_vs_meantask.pdf".format(prefix))


# each inducing task in 1 plot
fig, ax = plt.subplots(figsize=(2.5,2), tight_layout=True)

sns.kdeplot(mse_tasks_all, ax=ax, shade=True)
ax.get_lines()[0].remove()

for i in range(n_inducing_tasks):
    for j in range(i,n_inducing_tasks-1):
        sns.kdeplot(mse_u2u_all[i,j,:], ax=ax, label=r"t$\rm _{}$ - t$\rm _{}$".format(i,j+1), legend=True)

if savefig:
    fig.savefig("img/{}_dist_induces.pdf".format(prefix))


if not savefig:
    plt.show()
