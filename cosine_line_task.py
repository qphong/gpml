import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp 


xdim = 1
ydim = 1
ntheta = 5
xmin = -5.0
xmax = 5.0
"""
first 2 theta: amplitude and phase of cosine function
next 2 theta: intercept and slope of line
last theta: indicator: 1 if cosine function
                       0 if line function
"""


def get_random_datasets(ndataset, dataset_size):
    """
    theta[0]: amplitude [0.1, 5.0]
    theta[1]: phase     [0., pi]
    """
    X = np.random.rand(1,ndataset,dataset_size,xdim) * (xmax - xmin) + xmin
    # [-5., 5.]

    theta = np.random.uniform(
                low=0., high=1., 
                size=(1, ndataset, ntheta))
    theta[:,:,0] = theta[:,:,0] * 4.9 + 0.1 # [0.1,5.0]
    theta[:,:,1] = theta[:,:,1] * np.pi # [0.0, pi]
    theta[:,:,2] = theta[:,:,2] * 6. - 3. # [-3.,3.]
    theta[:,:,3] = theta[:,:,3] * 6. - 3. # [-3.,3.]

    # randomly assign cosine and line with equal probability
    theta[:,:,4] = np.random.randint(low=0, high=2, size=(1,ndataset)) 

    Y = predict_np(X, theta)
    # (1,ndataset,dataset_size,ydim)

    X = X[0,...]
    # (ndataset,dataset_size,xdim)
    Y = Y[0,...]
    # (ndataset,dataset_size,ydim)

    return X, Y, theta.reshape(ndataset, ntheta)


def get_y_distance(y0, y1):
    # y0: (...,ntask0, ntask1|1, npoint, ydim)
    # y1: (...,ntask0|1, ntask1, npoint, ydim)
    return tf.reduce_mean(
                    (y0 - y1) * (y0 - y1),
                    axis = -1)
    # (...,ntask0, ntask1, npoint)


def eval_loss(theta, Xtrain, Ytrain, debug=False):
    # theta: (n,m|1,ntheta)
    # Xtrain: (n|1,m,npoint,xdim)
    # Ytrain: (n|1,m,npoint,ydim)
    
    predicted_Y = predict(Xtrain,theta)
    # (n,m,npoint,1)

    # mean squared error:
    mse_val = tf.reduce_mean(
        tf.reduce_mean( (Ytrain - predicted_Y) * (Ytrain - predicted_Y), axis= 2),
        axis=2)
    # (n,m)
    
    if debug:
        return mse_val, predicted_Y
        
    return mse_val


def predict(x, theta):
    # theta: (n,m|1,ntheta)
    # x: (n|1,m,npoint, xdim)
    # returns: (n,m,npoint, ydim)
    
    theta = tf.expand_dims(theta, axis=-2)
    # (...,1,ntheta)

    y_cosine = tf.expand_dims(theta[...,0] * tf.cos(theta[...,1] + x[...,0]), axis=-1)
    # (n,m,npoint,1)

    y_line = tf.expand_dims(theta[...,2] * x[...,0] + theta[...,3], axis=-1)
    # (n,m,npoint,1)

    y = y_cosine * theta[...,4:5] + y_line * (1. - theta[...,4:5])
    # (n,m,npoint,1)

    return y


def predict_np(x, theta):
    # theta: (n,m|1,ntheta)
    # x: (n|1,m,npoint, xdim)
    # returns: (n,m,npoint, ydim)
    
    theta = np.expand_dims(theta, axis=-2)
    # (...,1,ntheta)

    y_cosine = np.expand_dims(theta[...,0] * np.cos(theta[...,1] + x[...,0]), axis=-1)
    # (n,m,npoint,1)
    
    y_line = np.expand_dims(theta[...,2] * x[...,0] + theta[...,3], axis=-1)
    # (n,m,npoint,1)

    y = y_cosine * theta[...,4:5] + y_line * (1. - theta[...,4:5])
    # (n,m,npoint,1)

    return y


