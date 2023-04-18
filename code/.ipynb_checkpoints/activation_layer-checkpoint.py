import numpy as np
from layer import Layer

class activationLayer(Layer):
    def __init__(self ,activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward_prop(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    def back_prop(self, E_y, E_y2, eta, optim_method, search_method, temp, linesearch, rand_low, rand_up, K, tolerate, \
                tolerate_g, tolerate_distance, cg_formula, g_w = None, g_b = None, H_w = None, H_b = None, pk = None, j = None, index = None, partial = None, l = None):
        # print('E_y', E_y, E_y.shape)
        # print('self.input', self.input, self.input.shape)
        # print('self.activation_prime(self.input),', self.activation_prime(self.input).shape)
        # print('E_y2', E_y2, E_y2.shape)
        g_w = self.activation_prime(self.input) * g_w
        g_b = self.activation_prime(self.input) * g_b
        if (optim_method == 'steepest') | (optim_method == 'conjugate'):
            H_w = np.matmul(np.matmul(self.input.T, E_y2 * (self.activation_prime(self.input) * self.activation_prime(self.input)).T), self.input)
            H_b = E_y2
        return self.activation_prime(self.input) * E_y, g_w, g_b, H_w, H_b