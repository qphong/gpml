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

from gpml2 import GPML 

import cosine_task
import cosine_line_task
import polynomial_task 
import noisy_sine_task

from NetApproximator import NetApproximator 


task_name = "cosine"
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


n_inducing_tasks = 4
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
    n_predicted_param_sample = 100,

    is_pivot_X_adaptive_in_Kuu = is_pivot_X_adaptive_in_Kuu,
    pivot_X = pivot_X, # (nXu, xdim)
    
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


n_test_tasks = metalearn.training_task_batchsize
n_test_datapoints = 1000

X_np, Y_np, test_params_np = task.get_random_datasets(
    n_test_tasks,
    metalearn.training_task_datasize + n_test_datapoints
)

X_train_np = X_np[:,:metalearn.training_task_datasize,:]
Y_train_np = Y_np[:,:metalearn.training_task_datasize,:]
X_test_np = X_np[:,metalearn.training_task_datasize:,:]
Y_test_np = Y_np[:,metalearn.training_task_datasize:,:]


# for visualization we require xdim = 1
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


mean_theta_np, std_theta_np, theta_samples_np, updated_theta_samples_np \
    = metalearn.predict_param(
        X_train_np, 
        Y_train_np, 
        learned_params)
# theta_samples_np: (n_theta_sample, ndataset, ntheta)


with tf.Session(graph=graph) as sess:

    theta_y_np = []
    for i in range(learned_params['theta_u'].shape[0]):
        theta_y_np.append(
            sess.run(predicted_y,
                        feed_dict = {
                            x_plc: np.expand_dims(
                                    X_test_np, axis=0) ,
                            param_plc: learned_params['theta_u'][i,...].reshape(1,1,netapprox.n_param)
                        })
        )
    
    mean_y_np = sess.run(predicted_y,
                        feed_dict = {
                            x_plc: np.expand_dims(
                                    X_test_np, axis=0) ,
                            param_plc: learned_params['scalar_mean'].reshape(1,1,netapprox.n_param)
                        })

    predicted_y_np = sess.run(predicted_y,
                        feed_dict = {
                            x_plc: np.expand_dims(
                                    X_test_np, axis=0) ,
                            param_plc: theta_samples_np
                        })

    predicted_updated_y_np = sess.run(predicted_y,
                        feed_dict = {
                            x_plc: np.expand_dims(
                                    X_test_np, axis=0) ,
                            param_plc: updated_theta_samples_np
                        })


print("Average MSE:", np.mean( np.square( predicted_y_np - Y_test_np ) ))
print("            ", np.mean( np.square( predicted_updated_y_np - Y_test_np ) ))


"""
3 plots:
    1. plot of predicted_y with all theta_samples (shape = (n_task_plot, n_theta_sample_plot))
    2. plot of predicted_y with inducing tasks (shape = (n_inducing_task,))
    3. plot of the predicted_y with the mean_task (shape = (1,))
"""

n_task_plot = np.min([X_test_np.shape[0], 5])
n_theta_sample_plot = np.min([predicted_y_np.shape[0], 30])

selected_idxs = np.array(list(range(n_task_plot)))
# selected_idxs = [2, 3, 4]

# plot all theta and each task in 1 subplot
fig, axs = plt.subplots(1, len(selected_idxs), 
                figsize=(len(selected_idxs) * 1.5 * 1.1, 1 * 2),
                sharex=True,
                sharey=True)

fig.tight_layout(rect=(0.0, 0.0, 1., 0.9))
# origin at bottom left


axs = axs.reshape(1, len(selected_idxs))

for idx_param in range(n_theta_sample_plot):

    y_min = 1e10
    y_max = -1e10

    for idx_task in selected_idxs:

        y_min = np.min([y_min, 
                    Y_test_np[idx_task,:,:].min(),
                    Y_train_np[idx_task,...].min()])
        y_max = np.max([y_max, 
                    Y_test_np[idx_task,:,:].max(),
                    Y_train_np[idx_task,...].max()])

    y_max += (y_max - y_min) / 6
    y_min -= (y_max - y_min) / 6

    for i,idx_task in enumerate(selected_idxs):

        if idx_param == 0:
            axs[0,i].plot(
                            X_test_np[idx_task,...].squeeze(),
                            Y_test_np[idx_task,:,:].squeeze(), 
                            c = colors[0],
                            label=r'Ground truth')

        axs[0,i].plot(
                        X_test_np[idx_task,...].squeeze(),
                        predicted_y_np[idx_param,idx_task,:,:].squeeze(), 
                        '--',
                        c = colors[1],
                        alpha=0.1 if n_theta_sample_plot > 10 else 1.0)

        axs[0,i].plot(
                        X_test_np[idx_task,...].squeeze(),
                        predicted_updated_y_np[idx_param,idx_task,:,:].squeeze(), 
                        ':',
                        c = colors[2],
                        alpha=0.1 if n_theta_sample_plot > 10 else 1.0)

        # Plot non-displayed NaN line for legend, leave alpha at default of 1.0
        axs[0,i].plot( np.NaN, np.NaN, '--', c=colors[1], alpha=1.0, label=r'0 gradient step' )
        axs[0,i].plot( np.NaN, np.NaN, ':', c=colors[2], alpha=1.0, label=r'1 gradient step' )
        
        if idx_param == n_theta_sample_plot - 1:
            axs[0,i].scatter(
                            X_train_np[idx_task,...].squeeze(),
                            Y_train_np[idx_task,...].squeeze())

        axs[0,i].set_ylim(y_min, y_max)

        if len(selected_idxs) == 5:
            if (idx_param == 0 and idx_task == 1):
                axs[0,idx_task].legend(
                    bbox_to_anchor=(0.27, 1.02, 
                        3.3, .102), 
                    loc=3,
                    ncol=3, 
                    mode="expand", 
                    borderaxespad=0)
        else:

            if (idx_param == 0 and i == 0):
                axs[0,i].legend(
                    bbox_to_anchor=(0.05, 1.02, 
                        3.2, .102), 
                    loc=3,
                    ncol=3, 
                    mode="expand", 
                    borderaxespad=0)




prefix = "img/{}_gml{}_{}_u{}_batch{}_k{}_nstep{}".format(
    task_name,
    "Bayes" if use_samples else "",
    "adt" if is_pivot_X_adaptive_in_Kuu else "fix",
    n_inducing_tasks,
    training_task_batchsize,
    training_task_datasize,
    n_step)

if savefig:
    fig.savefig("{}_samples_{}_{}.pdf".format(prefix, selected_idxs[0], len(selected_idxs)))



n_inducing_plot_row = 1
n_inducing_plot_col = int(np.ceil(n_inducing_tasks/n_inducing_plot_row))

fig, axs = plt.subplots(n_inducing_plot_row, n_inducing_plot_col,
            figsize = (n_inducing_plot_col * 2, n_inducing_plot_row * 2),
            sharex=True,
            sharey=True)

axs = axs.reshape(n_inducing_plot_row, n_inducing_plot_col)

fig.tight_layout()
# origin at bottom left

y_max = -1e10
y_min = 1e10
for i in range(n_inducing_tasks):
    y_max = np.max([y_max, theta_y_np[i][0,0,...].max()])
    y_min = np.min([y_min, theta_y_np[i][0,0,...].min()])
    
for i in range(n_inducing_tasks):
    c,r = int(i / n_inducing_plot_col), i % n_inducing_plot_col

    axs[c,r].plot(X_test_np[0,...].squeeze(),
            theta_y_np[i][0, 0,...].squeeze())

    axs[c,r].set_ylim(y_min, y_max)

if savefig:
    fig.savefig("{}_inducing{}.pdf".format(
        prefix,
        "" if n_inducing_plot_row == 1 else "_{}x{}".format(n_inducing_plot_row, n_inducing_plot_col)
        ))



# plot inducing without sharey
fig, axs = plt.subplots(1, n_inducing_tasks,
            figsize = (n_inducing_tasks * 1.5, 1.5),
            sharex=True)
axs = axs.reshape(1,-1)

fig.tight_layout()
# origin at bottom left

y_max = -1e10
y_min = 1e10
for i in range(n_inducing_tasks):
    y_max = np.max([y_max, theta_y_np[i][0,0,...].max()])
    y_min = np.min([y_min, theta_y_np[i][0,0,...].min()])
    
for i in range(n_inducing_tasks):
    axs[0,i].plot(X_test_np[0,...].squeeze(),
            theta_y_np[i][0, 0,...].squeeze())

if savefig:
    fig.savefig("{}_inducing_diffy.pdf".format(prefix))



fig, ax = plt.subplots(figsize=(1.5,1.5))

ax.plot(X_test_np[0,...].squeeze(),
        mean_y_np[0,0,:,:].squeeze(), label="mean")

fig.tight_layout()



if savefig:
    fig.savefig("{}_mean.pdf".format(prefix))

if not savefig:
    plt.show()


