import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np

clip_min = 1e-100

def compute_dmm(theta0, Xu, predict, get_y_distance):
    # theta0: (ntask,nXu|1,ntheta)
    # Xu: (ntask|1,nXu,npoint,xdim) # (npoint,xdim)
    # return (nXu,ntask,ntask,npoint)
    """
    for different datasets, we use different Xu
        so ntask2 is the number of datasets
    Xu is the same for all ntheta
    """

    y0 = predict( Xu, theta0 )
    # (ntask,nXu,npoint,ydim)

    y0 = tf.transpose(y0, perm=[1,0,2,3])
    # (nXu,ntask,npoint,ydim)

    y1 = tf.expand_dims(y0, axis=2)
    # (nXu,ntask, 1, npoint, ydim)
    y2 = tf.expand_dims(y0, axis=1)
    # (nXu, 1, ntask, npoint, ydim)

    d = get_y_distance(y1,y2)
    # (nXu,ntask,ntask,npoint)
    return d 


def compute_dmn(theta0, X, Y, predict, get_y_distance):
    # theta0: (ntask,ndataset|1, ntheta)
    # X: (ntask|1, ndataset, npoint, xdim)
    # Y: (ntask|1, ndataset, npoint, ydim)
    
    y0 = predict(X,theta0)
    # (ntask, ndataset, npoint, ydim)

    d = get_y_distance(y0, Y)
    # (ntask, ndataset, npoint)
    return d


def compute_K(dist, lscale, sigma):
    # dist: (...,npoint)
    # lscale: (ntheta,)
    # sigma: (ntheta,)
    # return K: (ntheta,...)

    dist = tf.expand_dims(dist, axis=-1)
    # (...,npoint,1)
    
    lscale = tf.squeeze(lscale)
    sigma = tf.squeeze(sigma)

    k = sigma * tf.exp( 
            - 0.5 * tf.reduce_mean(dist / tf.square(lscale),
                                  axis=-2) )
    # (...,ntheta)

    return k


def compute_mean_var_theta(Xtest, Ytest, 
                theta_u,
                scalar_mean,
                predict, get_y_distance,
                lscale, sigma, 
                invKu):
    """
    Xtest: (ntest,npoint,xdim)
    Ytest: (ntest,npoint,ydim)
    theta_u: (nu,ntheta)
    scalar_mean: (ntheta)
    lscale: (ntheta,)
    sigma: (ntheta,)
    invKu: (ntheta,ntest,nu,nu)
    """

    dist_test = compute_dmn(
                tf.expand_dims(theta_u, axis=1), 
                tf.expand_dims(Xtest, axis=0), 
                tf.expand_dims(Ytest, axis=0), 
                predict, get_y_distance)
    # (nu, ntest, npoint)
    kutest = compute_K(dist_test, lscale, sigma)
    # (nu, ntest, ntheta)

    ktestu = tf.transpose(kutest, perm=[2,1,0])
    # (ntheta, ntest, nu)

    theta_u_minus_mean = theta_u - tf.expand_dims(scalar_mean, axis=0)
    theta_u_minus_mean = tf.transpose(theta_u_minus_mean)
    theta_u_minus_mean = tf.expand_dims(theta_u_minus_mean, axis=2)
    # (ntheta,nu,1)
    theta_u_minus_mean = tf.expand_dims(theta_u_minus_mean, axis=1)
    # (ntheta,1,nu,1)

    # invKu: (ntheta,ntest,nu,nu)

    tmp = tf.reduce_mean(invKu @ theta_u_minus_mean, axis=-1)
    # (ntheta,ntest,nu)
    mean = tf.reduce_sum(ktestu * tmp, axis=2)
    # (ntheta,ntest)


    sigma = tf.expand_dims(sigma, axis=1)
    # (ntheta,1)
    ktestu = tf.expand_dims(ktestu, axis=2)
    # (ntheta,ntest,1,nu)
    tmp = ktestu @ invKu
    # (ntheta,ntest,1,nu)


    var = sigma - tf.reduce_sum(
                    tf.reduce_sum(tmp * ktestu, axis = -1),
                    axis=-1)
    # (ntheta,ntest)

    var = tf.clip_by_value(var, clip_value_min=clip_min, clip_value_max=np.infty)
    return mean,var




