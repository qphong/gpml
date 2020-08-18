import os 

import numpy as np 
import tensorflow as tf 

import matplotlib.pyplot as plt 

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


from gpml2 import GPML 
from maml import MAML 


import cosine_task
import cosine_line_task
import polynomial_task 
import sensor_task
import noisy_sine_task

from NetApproximator import NetApproximator 



def get_mse(folder, n_total_tesk_task, n_test_datapoints, seed):
    path_mse = "{}/mse_ntask{}_npoint{}_seed{}.p".format(
                    folder, 
                    n_total_tesk_task, 
                    n_test_datapoints, 
                    seed)

    with open(path_mse, "rb") as readfile:
        data = pickle.load(readfile)
        mse = data['mse']
        updated_mse = data['updated_mse']

    return mse, updated_mse


def get_identifier(item):    
    if item['method'] == "maml":
        identifier = "maml_d{}_step{}".format(
            item['datasize'], item['step'])
    
    elif item['method'] in ["gml", "gpml"]:
        identifier = "gpml_d{}_step{}_u{}_bayes{}_adapt{}".format(
            item['datasize'], item['step'], 
            item['inducing'], item['Bayes'], 
            item['adaptiveu'])

    return identifier


def get_crit_info(item, legend_type, color_type):
    color_dict = {
        "method": {
            "MAML": colors[0],
            "GPML-Bayes": colors[1],
            "GPML-lite": colors[2],
            "GPML": colors[3]
        },
        "datasize": {
            0: colors[0],
            5: colors[1],
            15: colors[2],
            20: colors[3],
            25: colors[4]
        },
        "inducing": {
            2: colors[0],
            4: colors[1],
            6: colors[2],
            8: colors[3],
            10: colors[4],
            16: colors[5]
        }
    }

    method = item['method']
    if method in ['gpml', 'gml', 'GPML', 'GML']:
        if not item['adaptiveu']:
            method = 'GPML-lite'
        else:

            if item['Bayes']:
                method = 'GPML-Bayes'
            else:
                method = 'GPML'
    else:
        method = "MAML"
    
    legend = ""

    for t in legend_type:

        legend = legend if len(legend) == 0 else "{} ".format(legend)

        if t == "method":
            legend = "{}{}".format(legend, method)

        elif t == "datasize":
            legend = "{}k={}".format(legend, item['datasize'])

        elif t == "inducing" and item['method'] in ['gpml', 'gml']:
            legend = r"{}\rm $n_u$={}".format(legend, item['inducing'])

        elif t == "Bayes" and item['method'] in ['gpml', 'gml']:
            # already in method
            pass

        elif t == "adaptiveu" and item['method'] in ['gpml', 'gml']:
            # already in method
            pass

        elif t == "step":
            legend = "{}m={}".format(legend, item['step'])


    color_feature = 0
    if color_type == "method":
        color_feature = method
    elif color_type == "datasize":
        color_feature = item['datasize']
    elif color_type == "inducing":
        color_feature = item['inducing']

    return {"legend": legend, "color": color_dict[color_type][color_feature]}



def plot(ax, measures, items, title, legend_type, color_type, xlogscale=False, show_outliers=False):
    # dictionary of crit_name to its mse or updated_mse

    names = np.array( list(mse_all.keys()) )

    min_len = np.min([measures[n].shape[0] for n in names])

    mse = np.stack([measures[n][:min_len] for n in names])
    # n_names,n_test_tasks

    avg_mse = np.mean(mse, axis=1)
    # n_names
    if verticle:
        sorted_idxs = np.argsort(-avg_mse)
    else:
        sorted_idxs = np.argsort(avg_mse)

    names = names[sorted_idxs]
    mse = mse[sorted_idxs,:].T

    plot_notations = [get_crit_info(items[n], legend_type, color_type) for n in names]


    medianprops = dict(color='#2F2F2F') # dict(linestyle='-.', linewidth=2.5, color='firebrick')
    meanpointprops = dict(marker='D', markeredgecolor='black',
                        markerfacecolor='firebrick')
    flierprops = dict(marker='o', markerfacecolor='#5F5F5F', markeredgecolor='#5F5F5F', markersize=5,
        linestyle="None")

    bp = ax.boxplot(mse, flierprops=flierprops, vert=verticle, patch_artist=True, showfliers=show_outliers)

    for element in ['medians']:#, 'whiskers', 'fliers', 'means', 'caps']:
        for cidx,c in enumerate(names):
            plt.setp(bp[element][cidx], color='#2F2F2F')

    for cidx,c in enumerate(names):
        bp['boxes'][cidx].set(facecolor=plot_notations[cidx]['color']) 


    if not verticle:
        if xlogscale:
            ax.set_xscale('log')

        ax.set_yticklabels([notation['legend'] for notation in plot_notations])
        ax.set_xticks([0.01, 0.1])

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14) 
            
    else:
        if xlogscale:
            ax.set_yscale('log')

        ax.set_xticklabels([notation['legend'] for notation in plot_notations],
            rotation=70)
     


task_name = "polynomial"
xlogscale = True
show_outliers = False
verticle = False

if task_name == "cosine":
    task = cosine_task
elif task_name == "cosineline":
    task = cosine_line_task
elif task_name == "polynomial":
    task = polynomial_task 
elif  task_name == 'sensorlight':
    task = sensor_task.Sensor('light')
else:
    raise Exception("Unknown task: {}".format(task_name))


np.random.seed(0)
tf.set_random_seed(1)


dtype = tf.float64 
xdim = task.xdim
ydim = task.ydim

# "method", "datasize", "inducing", "Bayes", "step"
"""
example:
    method: gml maml
    datasize: 5, 10, 15
    inducing: 2,4,6,8
    Bayes: 0,1
    step:0,1,5
"""


# sensor dataset
method = ["maml", "gpml"]
n_inducing_tasks = [8]
training_task_batchsize = 5
training_task_datasize = [5]
is_pivot_X_adaptive_in_Kuu = [True]
n_step = 1
use_samples = False
n_pivot_X = 100 # used to c


color_type = "method"
legend_type = ["method", "inducing", "step"]#["method", "datasize", "inducing", "step"]
fig_title_postfix = "compare_nstep1"

if fig_title_postfix == "compare_nstep1":
        
    # cosine
    if task_name == "cosine":
        plot_items = [
                {"method": "maml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "maml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 5},
                {"method": "maml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 10},
                {"method": "gml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "gml",
                    "inducing": 4,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "gml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": False,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "gml",
                    "inducing": 4,
                    "datasize": 5,
                    "Bayes": False,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "gml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": False,
                    "adaptiveu": False,
                    "step": 1},
                {"method": "gml",
                    "inducing": 4,
                    "datasize": 5,
                    "Bayes": False,
                    "adaptiveu": False,
                    "step": 1}
            ]



    # cosineline
    elif task_name == "cosineline":
        plot_items = [
                {"method": "maml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "maml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 5},
                {"method": "maml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 10},
                {"method": "gml",
                    "inducing": 4,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "gml",
                    "inducing": 8,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "gml",
                    "inducing": 4,
                    "datasize": 5,
                    "Bayes": False,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "gml",
                    "inducing": 8,
                    "datasize": 5,
                    "Bayes": False,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "gml",
                    "inducing": 4,
                    "datasize": 5,
                    "Bayes": False,
                    "adaptiveu": False,
                    "step": 1},
                {"method": "gml",
                    "inducing": 8,
                    "datasize": 5,
                    "Bayes": False,
                    "adaptiveu": False,
                    "step": 1}
            ]


    elif task_name == "polynomial":
        plot_items = [
                {"method": "maml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "maml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 5},
                {"method": "gml",
                    "inducing": 4,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "gml",
                    "inducing": 8,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 1}
            ]

    # sensorlight
    elif task_name == "sensorlight":

        plot_items = [
                {"method": "maml",
                    "inducing": 8,
                    "datasize": 10,
                    "Bayes": False,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "maml",
                    "inducing": 8,
                    "datasize": 10,
                    "Bayes": False,
                    "adaptiveu": True,
                    "step": 5},
                {"method": "maml",
                    "inducing": 8,
                    "datasize": 10,
                    "Bayes": False,
                    "adaptiveu": True,
                    "step": 10},
                {"method": "gml",
                    "inducing": 8,
                    "datasize": 10,
                    "Bayes": False,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "gml",
                    "inducing": 20,
                    "datasize": 10,
                    "Bayes": False,
                    "adaptiveu": True,
                    "step": 1}
            ]

elif fig_title_postfix == "compare_nstep0":

    # cosine
    if task_name == "cosine":
        plot_items = [
                {"method": "maml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "maml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 5},
                {"method": "maml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 10},
                {"method": "gml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": False,
                    "adaptiveu": True,
                    "step": 0},
                {"method": "gml",
                    "inducing": 4,
                    "datasize": 5,
                    "Bayes": False,
                    "adaptiveu": True,
                    "step": 0}
            ]


    # cosineline
    elif task_name == "cosineline":
        plot_items = [
                {"method": "maml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 1},
                {"method": "maml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 5},
                {"method": "maml",
                    "inducing": 2,
                    "datasize": 5,
                    "Bayes": True,
                    "adaptiveu": True,
                    "step": 10},
                {"method": "gml",
                    "inducing": 4,
                    "datasize": 5,
                    "Bayes": False,
                    "adaptiveu": True,
                    "step": 0},
                {"method": "gml",
                    "inducing": 8,
                    "datasize": 5,
                    "Bayes": False,
                    "adaptiveu": True,
                    "step": 0}
            ]


seed = 0
np.random.seed(seed)

if task_name.startswith("sensor"):
    n_total_tesk_task = 1000
else:
    n_total_tesk_task = 100

n_test_tasks = training_task_batchsize
n_repetitions = int(n_total_tesk_task / n_test_tasks)
n_total_tesk_task = n_repetitions * n_test_tasks

n_test_datapoints = 1000

mse_all = {}
updated_mse_all = {}
item_dict = {}

for item in plot_items:

    method_i = item['method']
    training_task_datasize_i = item['datasize']
    n_step = item['step']
    n_inducing_tasks_i = item['inducing']
    use_samples = item['Bayes']
    is_pivot_X_adaptive_in_Kuu_i = item['adaptiveu']

    identifier = get_identifier(item)

    if method_i == "maml":

        folder = "{}/maml_batch{}_k{}_nstep{}".format(task_name, 
                training_task_batchsize,
                training_task_datasize_i,
                n_step)

        mse, updated_mse = get_mse(folder, n_total_tesk_task, n_test_datapoints, seed)

    else:

        folder = "{}/gml{}_{}_u{}_batch{}_k{}_nstep{}".format(
                    task_name,
                    "Bayes" if use_samples else "",
                    "adt" if is_pivot_X_adaptive_in_Kuu_i else "fix",
                    n_inducing_tasks_i,
                    training_task_batchsize,
                    training_task_datasize_i,
                    n_step
                )

        mse, updated_mse = get_mse(folder, n_total_tesk_task, n_test_datapoints, seed)

    mse_all[identifier] = mse 
    updated_mse_all[identifier] = updated_mse
    item_dict[identifier] = item



prefix = "{}_{}".format(task_name, "log_" if xlogscale else "")
prefix = "{}{}".format(prefix, "outlier_" if show_outliers else "")
prefix = "{}{}".format(prefix, "vert_" if verticle else "")

# fig_title_postfix
prefix = "{}{}_".format(prefix, fig_title_postfix)


figsize = (0.4 * len(legend_type) + 2.4, len(item_dict)*2.5/5 - 0.8)
fig, ax = plt.subplots(figsize=figsize)

plot(ax, mse_all, item_dict, "MSE", legend_type, color_type, xlogscale, show_outliers)
fig.tight_layout()

fig.savefig("img/{}mse.pdf".format(prefix))


fig, ax = plt.subplots(figsize=figsize)
plot(ax, updated_mse_all, item_dict, "Updated MSE", legend_type, color_type, xlogscale, show_outliers)
fig.tight_layout()

fig.savefig("img/{}updated_mse.pdf".format(prefix))
