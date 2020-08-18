import numpy as np 
import tensorflow as tf 


class NetApproximator():

    def __init__(self, xdim, layer_sizes, activations): 
        assert len(layer_sizes) == len(activations), \
            "The length of layer_sizes and activations must equal!"

        self.xdim = xdim 
        self.nlayer = len(layer_sizes)
        self.layer_sizes = layer_sizes 
        self.activations = activations

        n_param_per_layer = np.zeros(self.nlayer, dtype=int)

        n_param_per_layer[0] = (xdim + 1) * layer_sizes[0]
        for i in range(self.nlayer-1):
            n_param_per_layer[i+1] = (layer_sizes[i]+1) * layer_sizes[i+1]

        cs_n_param_per_layer = np.cumsum(n_param_per_layer)

        self.n_param_per_layer = n_param_per_layer 
        self.n_param = cs_n_param_per_layer[-1]
        self.cs_n_param_per_layer = cs_n_param_per_layer


    @staticmethod
    def make_activation_layer(x, activation):
        if activation == 'relu':
            network = tf.nn.relu(x)
        elif activation == 'leaky_relu':
            network = tf.nn.leaky_relu(x, alpha=0.2)
        elif activation == 'sigmoid':
            network = tf.nn.sigmoid(x)
        elif activation == 'linear':
            network = x
        else:
            raise Exception("Only allow relu, leaky_relu, sigmoid, linear. Unknown activation: {}".format(activation))
        
        return network


    def predict(self, x, 
            params):
        # nlayer: scalar
        # layer_sizes: list of size nlayer
        # activations: list of activations
        # x:      (n|1, m, npoint, xdim)
        # params: (n, m|1, n_param)
        #     {'relu', 'leaky_relu', 'sigmoid'}
        """
        nlayer = 2
        layer_size = [2,3]
        network: xdim -> 2 -> 3
        (output_dim = 3)
        returns:
            network
            params: list of tensor of shape:
                [xdim+1,layer_size[0]]
                [layer_size[i]+1, layer_size[i+1]]
        """
        xdim = self.xdim 
        nlayer = self.nlayer 
        layer_sizes = self.layer_sizes
        activations = self.activations 

        cs_n_param_per_layer = self.cs_n_param_per_layer

        param_0 = tf.gather(params, 
                    indices=list(range(cs_n_param_per_layer[0])),
                    axis=2)
        param_0 = tf.reshape(param_0, 
                    shape=(tf.shape(param_0)[0], 
                        tf.shape(param_0)[1], 
                        xdim+1, 
                        layer_sizes[0]))

        W = tf.gather(param_0, indices=list(range(xdim)), axis=2)
        # (n, m|1, xdim, layer_sizes[0])
        b = tf.gather(param_0, indices=[xdim], axis=2)
        # (n, m|1, 1,    layer_sizes[0])

        net = NetApproximator.make_activation_layer(
                        x @ W + b, 
                        activations[0])
                    
        for i in range(nlayer-1):
            param_i = tf.gather(params, 
                        indices=list(range(cs_n_param_per_layer[i], cs_n_param_per_layer[i+1])),
                        axis=2)
            param_i = tf.reshape(param_i, 
                        shape=(tf.shape(param_i)[0],
                            tf.shape(param_i)[1], 
                            layer_sizes[i]+1, 
                            layer_sizes[i+1]))
            W = tf.gather(param_i, 
                        indices=list(range(layer_sizes[i])),
                        axis=2)
            # (n, m|1, layer_sizes[i], layer_sizes[i+1])
            b = tf.gather(param_i,
                        indices=[layer_sizes[i]], 
                        axis=2)
            # (n, m|1, 1,              layer_sizes[i+1])

            net = NetApproximator.make_activation_layer(
                            net @ W + b, 
                            activations[i+1])
                                    
        return net
