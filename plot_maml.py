import os 


import numpy as np 
import tensorflow as tf 


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

from maml import MAML

import cosine_task
import cosine_line_task
import polynomial_task 
import noisy_sine_task 

from NetApproximator import NetApproximator 


task_name = "polynomial"
savefig = True

if task_name == "cosine":
    task = cosine_task
elif task_name == "cosineline":
    task = cosine_line_task
elif task_name == "polynomial":
    task = polynomial_task 
elif task_name == "noisysine":
    task = noisy_sine_task


np.random.seed(0)
tf.set_random_seed(1)


dtype = tf.float64 
xdim = task.xdim
ydim = task.ydim


training_task_batchsize = 5
training_task_datasize = 5
n_step = 5


folder = "{}/maml_batch{}_k{}_nstep{}".format(task_name, 
            training_task_batchsize,
            training_task_datasize,
            n_step)

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


np.random.seed(2)


n_test_tasks = training_task_batchsize
n_test_datapoints = 1000

X_np, Y_np, test_params_np = task.get_random_datasets(
    n_test_tasks,
    training_task_datasize + n_test_datapoints
)

X_train_np = X_np[:,:training_task_datasize,:]
Y_train_np = Y_np[:,:training_task_datasize,:]
X_test_np = X_np[:,training_task_datasize:,:]
Y_test_np = Y_np[:,training_task_datasize:,:]


theta_samples_np, updated_theta_samples_np \
    = metalearn.predict_param(
        X_train_np, 
        Y_train_np, 
        learned_params)

theta_samples_np = np.tile(
    theta_samples_np.reshape(1,1,theta_samples_np.shape[-1]),
    reps=(1,updated_theta_samples_np.shape[1],1))


assert xdim == 1

# ntask,npoint,xdim (or ydim)
# sort both X_np and Y_np based on X_np in increasing order
for i in range(X_np.shape[0]):
    sorted_idxs = np.argsort(X_train_np[i,:,0])
    X_train_np[i,:,0] = X_train_np[i,sorted_idxs,0]
    Y_train_np[i,:,0] = Y_train_np[i,sorted_idxs,0]

    sorted_idxs = np.argsort(X_test_np[i,:,0])
    X_test_np[i,:,0] = X_test_np[i,sorted_idxs,0]
    Y_test_np[i,:,0] = Y_test_np[i,sorted_idxs,0]


with tf.Session(graph=graph) as sess:

    mean_y_np = sess.run(predicted_y,
                        feed_dict = {
                            x_plc: np.expand_dims(
                                    X_test_np, axis=0) ,
                            param_plc: theta_samples_np
                        })

    updated_mean_y_np = sess.run(predicted_y,
                        feed_dict = {
                            x_plc: np.expand_dims(
                                    X_test_np, axis=0) ,
                            param_plc: updated_theta_samples_np
                        })

"""
3 plots:
    1. plot of predicted_y with all theta_samples (shape = (n_task_plot, n_theta_sample_plot))
    2. plot of predicted_y with inducing tasks (shape = (n_inducing_task,))
    3. plot of the predicted_y with the mean_task (shape = (1,))
"""

prefix = "img/{}_maml_batch{}_k{}_nstep{}".format(task_name, 
            training_task_batchsize,
            training_task_datasize,
            n_step)

n_task_plot = np.min([X_test_np.shape[0], 5])

start_idx = 0 # to ignore the first start_idx tasks


fig, axs = plt.subplots(1, n_task_plot - start_idx, 
                figsize=((n_task_plot - start_idx) * 1.5 * 1.1, 1 * 2),
                sharex=True,
                sharey=True)

fig.tight_layout(rect=(0.0, 0.0, 1., 0.9))
# origin at bottom left

axs = axs.reshape(1, n_task_plot - start_idx)


y_min = 1e10
y_max = -1e10

for idx_task in range(n_task_plot - start_idx):

    idx_task += start_idx

    y_min = np.min([y_min, 
                Y_test_np[idx_task,:,:].min(),
                # mean_y_np[0,idx_task,:,:].min(),
                Y_train_np[idx_task,...].min()])
    y_max = np.max([y_max, 
                Y_test_np[idx_task,:,:].max(),
                # mean_y_np[0,idx_task,:,:].max(),
                Y_train_np[idx_task,...].max()])


y_max += (y_max - y_min) / 6
y_min -= (y_max - y_min) / 6


for idx_task in range(n_task_plot - start_idx):

    idx_task += start_idx

    axs[0,idx_task - start_idx].plot(X_test_np[idx_task,...].squeeze(),
            Y_test_np[idx_task,:,:].squeeze(), 
            label=r'Ground truth')
                    
    axs[0,idx_task - start_idx].plot(X_test_np[0,...].squeeze(),
            mean_y_np[0,idx_task,:,:].squeeze(), 
            '--',
            label=r'0 gradient step')

    axs[0,idx_task - start_idx].plot(X_test_np[0,...].squeeze(),
            updated_mean_y_np[0,idx_task,:,:].squeeze(), 
            ':',
            label='{} gradient step'.format(n_step))

    axs[0,idx_task - start_idx].scatter(X_train_np[idx_task,...].squeeze(),
            Y_train_np[idx_task,...].squeeze())

    axs[0,idx_task - start_idx].set_ylim(y_min, y_max)
  
    if start_idx == 0:
        if idx_task == 1:
            axs[0, idx_task].legend(
                bbox_to_anchor=(0.27, 1.02, 
                        3.3, .102), 
                loc=3,
                ncol=3, 
                mode="expand", 
                borderaxespad=0)
    else:
        if idx_task - start_idx == 0:
            axs[0, idx_task - start_idx].legend(
                    bbox_to_anchor=(0.05, 1.02, 
                        3.2, .102), 
                    loc=3,
                    ncol=3, 
                    mode="expand", 
                    borderaxespad=0)

fig.tight_layout()



if savefig:
    if start_idx == 0:
        fig.savefig("{}_samples.pdf".format(prefix))
    else:
        fig.savefig("{}_samples{}.pdf".format(prefix, start_idx))

if not savefig:
    plt.show()


