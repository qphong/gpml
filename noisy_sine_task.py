import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp 


xdim = 1
ydim = 1
ntheta = 3
xmin = -5.0 
xmax = 5.0

def get_random_datasets(ndataset, dataset_size):
    """
    A: theta[0]: amplitude [0.1, 5.0]
    b: theta[1]: phase     [0., 2*pi]
    w: theta[2]: frequency [0.5,2.0]
    epsilon ~ Normal(0, (0.01*A)^2)
    y = A sin(wx + b) + epsilon

    """
    X = np.random.rand(1,ndataset,dataset_size,xdim) * (xmax - xmin) + xmin
    # [-5., 5.]

    # theta = np.random.randn(1,ndataset, ntheta)
    theta = np.random.uniform(
                low=0., high=1., 
                size=(1, ndataset, ntheta))
    theta[:,:,0] = theta[:,:,0] * 4.9 + 0.1
    theta[:,:,1] = theta[:,:,1] * 2.0 * np.pi
    theta[:,:,2] = theta[:,:,2] * 1.5 + 0.5

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
    # (n,m|1,1,ntheta)

    noiseless_y = theta[...,0] * tf.sin(theta[...,1] + theta[...,2] * x[...,0])
    # (n,m,npoint)

    noise_std = 0.01 * theta[:,:,0:1,0] * tf.ones_like(noiseless_y)
    noise = tf.random.normal(shape=(1,), mean=tf.zeros_like(noise_std), stddev=noise_std)
    # (n,m,npoint)

    y = tf.expand_dims(noiseless_y + noise, axis=-1)
    # (n,m,npoint,1)

    return y


def predict_np(x, theta):
    # theta: (n,m|1,ntheta)
    # x: (n|1,m,npoint, xdim)
    # returns: (n,m,npoint, ydim)
    
    theta = np.expand_dims(theta, axis=-2)
    # (...,1,ntheta)

    noiseless_y = theta[...,0] * np.sin(theta[...,1] + theta[...,2] * x[...,0])
    # (n,m,npoint)

    noise = np.random.randn(noiseless_y.shape[0], noiseless_y.shape[1], noiseless_y.shape[2]) * 0.01 * theta[:,:,:,0]
    y = np.expand_dims(noiseless_y + noise, axis=-1)

    return y


