from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class WNN(Layer):
    """
    y_j = sum(w_i * Phi_i(x), i = 1..input_dim) + bias, j = 1..output_dim
    where Phi(x) = max(phi_h((x-t_h)/d_h)), h = 1..wavelon_count
    """
    def __init__(self, 
                 wavelon_count, 
                 output_dim,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.wavelon_count = wavelon_count
        self.output_dim = output_dim
        super(WNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dimensions = list(input_shape)[:-1:-1]
        self.w = self.add_weight(name='w',
                                 shape=(self.output_dim, self.wavelon_count),
                                 initializer = 'uniform',
                                 trainable=True)

        self.bias = self.add_weight(name='bias', 
                                 shape=(1, self.output_dim),
                                 initializer= 'uniform',
                                 trainable=True)

        self.dilation = self.add_weight(name='dilation', 
                                 shape=(self.wavelon_count, input_shape[-1]),
                                 initializer= 'uniform',
                                 trainable=True)

        self.translation = self.add_weight(name='translation', 
                                 shape=(self.wavelon_count, input_shape[-1]),
                                 initializer= 'uniform',
                                 trainable=True)

        super(WNN, self).build(input_shape)  

    def call(self, x):
        
        a_w = self.w
        a_b = self.bias
        a_d = self.dilation
        a_t = self.translation
        a_x = x #(,input_dim)
        a_x = K.repeat_elements(K.expand_dims(a_x, axis = -2), self.wavelon_count, -2) #(, wavelons, input_dim)

        for dim in self.input_dimensions:
            a_w = K.repeat_elements(K.expand_dims(a_w, 0), dim, 0) #(, output_dim, wavelon)
            a_b = K.repeat_elements(K.expand_dims(a_b, 0), dim, 0) #(, output_dim, 1)
            a_d = K.repeat_elements(K.expand_dims(a_d, 0), dim, 0) #(, wavelon, input_dim)
            a_t = K.repeat_elements(K.expand_dims(a_t, 0), dim, 0) #(, wavelon, input_dim)
        
        a_u = (a_x - a_t) / a_d #(, wavelons, input_dim)
        
        psi = -a_u * K.exp(-K.square(a_u) / 2.0)#(, wavelons, input_dim)
        psi = K.max(psi, axis=-1, keepdims=False)#(, wavelons)

        psi = K.repeat_elements(K.expand_dims(psi, axis = -2), self.output_dim, -2)#(, output_dim, wavelons)
        xc = K.sum(a_w * psi,axis=-1, keepdims=False) + a_b #(, output_dim)
        return xc
        
    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.output_dim,)
