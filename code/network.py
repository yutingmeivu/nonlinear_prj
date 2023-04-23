import numpy as np
import matplotlib.pyplot as plt

class NN:
    def __init__(self):
        # self.layers = []
        self.loss = None
        self.loss_prime = None
        self.loss_H = None
        self.weight = []
        self.bias = []
        self.input_layer = []
        self.weight_tmp = []
        self.bias_tmp = []
        self.layer = 0
        
    def load(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def activate(self, activation, activation_prime, activation_h):
        self.activation = activation
        self.activation_prime = activation_prime
        self.activation_h = activation_h
        
    def add(self, layer, type_):
        if type_ == 'fc':
            self.weight.append([np.random.rand(layer[0], layer[1])]) # D*N matrix
            self.bias.append([np.random.rand(1, layer[1])]) # 1*N vector
        else:
            # activation layer doesn't have w & b
            self.weight.append([None])
            self.bias.append([None])
        self.layer += 1
    
    def use(self, loss, loss_prime, loss_H):
        self.loss = loss
        self.loss_prime = loss_prime
        self.loss_H = loss_H
        
    def forward_prop(self, input_data, layer_index):
        self.input_layer.append(input_data) # X in each layer
        if self.weight[layer_index][0] is not None:
            self.output = np.dot(input_data, self.weight[layer_index]) + self.bias[layer_index]
        else:
            self.output = self.activation(input_data)
        return self.output
    
    def backward_prop(self, output_error, output_hessian, layer_index, step_size, pk, partial, linesearch, search_pk):
        # output_error = dE/dY
        if self.weight[layer_index][0] is not None:
            input_error = np.dot(output_error, self.weight[layer_index].T) # dE/dX = dE/dY * W.T
            weights_error = np.dot(self.input_layer[layer_index].T, output_error) # dE/dW = X.T dE/dY
            bias_error = output_error # dE/dB = dE/dY
            input_h = np.dot(output_hessian, self.weight[layer_index].T) + np.dot(output_error, output_hessian) # (d^2 E/dY^2) * (dY/dX)^T + (dE/dY) * (d^2 Y/dX^2)
            weight_h = np.dot(np.dot(self.input_layer[layer_index].T, output_hessian), self.input_layer[layer_index]) # d^2E/dW^2 = X^T d^2E/dY^2 X
            bias_h = output_hessian # d^2E/dB^2 = d^2E/dY^2
        else:
            input_error = self.activation_prime(self.input_layer[layer_index]) * output_error # dE/dX = dE/dY \odot \sigma'(X)
            weights_error = np.dot(self.input_layer[layer_index].T, output_error * self.activation_prime(self.input_layer[layer_index])) # dE/dW = X.T (dE/dY \odot sigma'(X))
            bias_error = output_error * self.activation_prime(self.input_layer[layer_index]) # dE/dB = dE/dY \odot sigma'(X)
            input_h = np.dot(output_hessian * self.activation_prime(self.input_layer[layer_index]), self.weight[layer_index].T) + output_error * self.activation_h(self.input_layer[layer_index]) # (d^2 E/ dY^2) * sigma'(X) * (dY/dX)^T + (dE/dY) * sigma''(X)
            sigma_prime = self.activation_prime(self.input_layer[layer_index])
            sigma_h = self.activation_h(self.input_layer[layer_index])
            weight_h = np.dot(self.input_layer[layer_index].T, (output_hessian * np.dot(sigma_prime.T , sigma_prime) + output_error * sigma_h))
            # X.T * (d^2 E/dY^2 \odot sigma'(Z)^2 + dE/dY \odot sigma''(Z))
            # weight_h = np.dot(self.input_layer[layer_index].T, (np.dot(self.activation_prime(self.input_layer[layer_index]) * output_hessian, np.dot(self.weight[layer_index].T, self.weight[layer_index])) + np.dot(output_error, self.activation_h(self.input_layer[layer_index]) * self.weight[layer_index]) )) # X.T * (sigma'(X) * (d^2 E/dY^2) * W.T^2 + (dE/dY) * (sigma''(X) * W.T))
            bias_h = output_hessian * self.activation_prime(self.input_layer[layer_index]) + output_error * self.activation_h(self.input_layer[layer_index]) # d^2E/dY^2 \odot sigma'(XW+ B) + dE/dY sigma''(XW + B)
        if (linesearch) | (search_pk):
            # step_size: step_size_w or step_size_b, same as pk
            # linesearch: True(don't update weight until get step size t), same as search_pk
            if optim_method == 'steepest':
                pk = - weights_error if partial == 'w' else - bias_error
            if partial == 'w':
                self.weight_tmp[layer_index] += step_size * pk
            else:
                self.bias_tmp[layer_index] += step_size * pk
        else:
            if (optim_method == 'gd') | (optim_method == 'steepest'):
                self.weight[layer_index] -= step_size[0] * weights_error
                self.bias[layer_index] -= step_size[1] * bias_error
                self.weight_tmp[layer_index] = self.weight[layer_index]
                self.bias_tmp[layer_index] = self.bias[layer_index]
            else:
                # step_size: (step_size_w, step_size_b), same as pk
                self.weight[layer_index] += step_size[0] * pk[0]
                self.bias[layer_index] += step_size[1] * pk[1]
                self.weight_tmp[layer_index] = self.weight[layer_index]
                self.bias_tmp[layer_index] = self.bias[layer_index]
        return input_error, input_h, weights_error, bias_error, weight_h, bias_h
    
    def fit(self, epochs, eta, optim_method, linesearch_method, rand_low, rand_up, K, tolerate, cg_formula, print_rate, plt_rate):
        samples = len(self.x_train)
        err_path = []
        
        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = self.x_train[j]
                for m in range(self.layer):
                    output = self.forward_prop(output, m)
                
                err += self.loss(y_train[j], output)
                output_error = self.loss_prime(y_train[j], output)
                output_hessian = self.loss_h(y_train[j], output)
                for n in reversed(range(self.layer)):
                    if optim_method == 'gd':
                        output_error, output_hessian, weights_error, bias_error, weight_h, bias_h = self.backward_prop(output_error, output_hessian, n, (eta, eta), None, None, False, False)
                    elif optim_method == 'steepest':
                        t_w = self.linesearch(linesearch_method, 'w', start_g, rand_low, rand_up, K, tolerate)
                        t_b = self.linesearch(linesearch_method, 'b', start_g, rand_low, rand_up, K, tolerate)
                        output_error, output_hessian, weights_error, bias_error, weight_h, bias_h = self.backward_prop(output_error, output_hessian, n, (t_w, t_b), None, None, False, False)
                    elif optim_method == 'conjugate':
                        t_w = self.linesearch(linesearch_method, 'w', start_g, rand_low, rand_up, K, tolerate)
                        t_b = self.linesearch(linesearch_method, 'w', start_g, rand_low, rand_up, K, tolerate)
                        pk_w = self.conjugate_gradient(linesearch_method, 'w', rand_low, rand_up, K, tolerate, cg_formula)
                        pk_b = self.conjugate_gradient(linesearch_method, 'b', rand_low, rand_up, K, tolerate, cg_formula)
                        output_error, output_hessian, weights_error, bias_error, weight_h, bias_h = self.backward_prop(output_error, output_hessian, n, (t_w, t_b), (pk_w, pk_b), None, False, False)
                
            err /= samples
            if print_rate:
                print('epoch %d/%d   error=%f' % (i+1, epochs, err))
            if plt_rate:
                err_path.append(err)
                plt.plot(err_path)
                plt.title(f'Loss with back propagation weight and bias update with {optim_method} using {search_method} line search')
                plt.xlabel('number of iterations')
                plt.show()
    
    @staticmethod
    def newton1d(t, partial, g, h): 
        t = t - np.matmul(np.linalg.inv(h), g) # since the f' and f'' of newton's method is actually the Jacobian matrix and hessian matrix of objective function, so used matrix manipulation
        return t       
            
    def linesearch(self, linesearch_method, partial, start_g, rand_low, rand_up, K, tolerate):
        # start_g: gradient of either weight and bias from fit
        if linesearch_method == 'newton':
            t_layers, dgtmp, gtmp, err = [0] * len(self.weight)
            for i in range(4):
                t_layers = update_linesearch(i, t_layers, start_g, partial, linesearch_method)
                self.weight_tmp = self.weight
                self.bias_tmp = self.bias
        else:
            t_layers0 = [0] * len(self.weight)
            rand = np.random.uniform(rand_low, rand_up)
            t_layers1 = [rand] * len(self.weight)
            t_tmp = [None] * len(self.weight)
            k = 0
            if linesearch_method == 'regula_falsi':
                dg0 = self.generate_startp(t_layers0, partial)
                dg1 = self.generate_startp(t_layers1, partial)
                for i in range(len(t_layers[0])):
                    while dg0[i] * dg1[i] >= 0:
                        rand = np.random.uniform(rand_low, rand_up)
                        t_layers1[i] = [rand] * len(self.weight)
                        dg1 = self.generate_startp(t_layers1, partial, k)
            g0 = start_g
            g1 = start_g
            gt = start_g
            dis = np.linalg.norm(t_layers0 - t_layers1)
            while (k < K) & (dis > tolerate):
                tmp, dg0, g0, err0 = self.subupdate(i, t_layers0, None, g0, partial, linesearch_method, optim_method)
                self.weight_tmp = self.weight
                self.bias_tmp = self.bias
                tmp, dg1, g1, err1 = self.subupdate(i, t_layers1, None, g1, partial, linesearch_method, optim_method)
                self.weight_tmp = self.weight
                self.bias_tmp = self.bias
                for i in range(len(dg0)):
                    t_tmp[i] = self.rf_sc(partial, t_layers0[i], t_layers1[i], g0[i], g1[i])
                tmp, dgt, gt, errt = self.subupdate(i, tmp, None, gt, partial, linesearch_method, optim_method)
                self.weight_tmp = self.weight
                self.bias_tmp = self.bias
                if linesearch_method == 'regula_falsi':
                    for i in range(len(t_layers[0])):
                        if dgt[i] * dg0[i] < 0:
                            t_layers1[i] = t_tmp[i]
                            g1[i] = gt[i]
                            err1 = errt
                        else:
                            t_layers0[i] = t_tmp[i]
                            g0[i] = gt[i]
                            err0 = errt
                elif linesearch_method == 'secant':
                    t_layers0 = t_layers1
                    t_layers1 = t_tmp
                dis = np.linalg.norm(t_layers0 - t_layers1)
                k += 1
                
            self.weight_tmp = self.weight
            self.bias_tmp = self.bias
            if linesearch_method == 'regula_falsi':
                t_layers = t_layers0 if err0 < err1 else t_layers1
            else:
                t_layers = t_layers1
        return t_layers # t_layers has forward order of moving step in each layers
    
    def conjugate_gradient(self, linesearch_method, partial, rand_low, rand_up, K, tolerate, cg_formula):
        k = 0
        restart = [False] * len(self.weight)
        n_step = [0] * len(self.weight)
        gk = weights_error if partial == 'w' else bias_error
        gk1 = gk
        restart = self.n_step_check(restart, k)
        pk = [-gk] * len(self.weight)
        prev_start = ['steepest'] * len(self.weight)
        while (np.linalg.norm(gk1) > tolerate) & (k < K):
            t = self.linesearch(self, linesearch_method, partial, gk1, rand_low, rand_up, K, tolerate)
            for i in range(len(self.weight)):
                if (k == 0) | (restart[i] == True):
                    t, dg_l, gk1, err = self.subupdate(k, t, pk, gk1, partial, linesearch_method, 'steepest')
                    pk[i] = -gk[i]
                    prev_step[i] = 'steepest'
                else:
                    pk[i], gk[i], gk1[i] = self.conjugate_gradient_pk(pk[i], gk[i], gk1[i], k, cg_formula, restart[i], prev_step[i])
                    prev_step[i] = 'conjugate'
                    t_layers, dg_l, gk1, err = self.subupdate(k, t, pk, gk1, partial, linesearch_method, optim_method)
            k += 1
        return pk
            
    
    def n_step_check(self, restart, k):
        for i in range(self.weight):
            if partial == 'w':
                param = self.weight[i]
            else:
                param = self.bias[i]
                n_step[i] = (param.shape[0] * param.shape[1]) - 1
                restart[i] = True if k >= n_step[i] else False
        return restart
    
    def generate_startp(self, t1, partial, k):
        detg_l = [None] * len(self.weight)
        for j in range(samples):
            output = self.x_train[j]
            for m in range(self.layer):
                if k == 0:
                    weights_error = 0
                if partial == 'w':
                    self.weight_tmp[m] -= t1[m] * weights_error
                else:
                    self.bias_tmp[m] -= t1[m] * bias_error
                output = self.forward_propagation(output, m)
                        
            err += self.loss(y_train[j], output)
            output_error = self.loss_prime(y_train[j], output)
            output_hessian = self.loss_h(y_train[j], output)
            for n in reversed(range(self.layer)):
                output_error, output_hessian, weights_error, bias_error, weight_h, bias_h = self.backward_prop(output_error, output_hessian, n, t1[n], pk, partial, True, False)
                detg_l[n] = np.linalg.det(weights_error) if partial == 'w' else np.linalg.det(bias_error)
        return detg_l
    
    def subupdate(self, k, t_layers, pk, start_g, partial, linesearch_method, optim_method):
        dg_l = [None] * len(self.weight)
        g_l = [None] * len(self.weight)
        for j in range(samples):
            output = self.x_train[j]
            for m in range(self.layer):
                if k == 0:
                    weights_error = start_g
                    bias_error = start_g
                if partial == 'w':
                    self.weight_tmp[m] -= t_layers[m] * weights_error
                else:
                    self.bias_tmp[m] -= t_layers[m] * bias_error
                output = self.forward_propagation(output, m)
                            
            err += self.loss(y_train[j], output)
            output_error = self.loss_prime(y_train[j], output)
            output_hessian = self.loss_h(y_train[j], output)
            if optim_method != 'conjugate':
                for n in reversed(range(self.layer)):
                    pk = weights_error if partial == 'w' else bias_error
                    output_error, output_hessian, weights_error, bias_error, weight_h, bias_h = self.backward_prop(output_error, output_hessian, n, t_layers[n], pk, partial, True, False)
                    if linesearch_method == 'newton':
                        h = weight_h if partial == 'w' else bias_h
                        g = weights_error if partial == 'w' else bias_error
                        t_layers[n] = self.newton1d(t_layers[n], partial, g, h)
                        g_l[n] = weights_error if partial == 'w' else bias_error
                    else:
                        dg_l[n] = np.linalg.det(weights_error) if partial == 'w' else np.linalg.det(bias_error)
                        g_l[n] = weights_error if partial == 'w' else bias_error
            else:
                for n in reversed(range(self.layer)):
                    output_error, output_hessian, weights_error, bias_error, weight_h, bias_h = self.backward_prop(output_error, output_hessian, n, t_layers[n], pk[n], partial, True, False)
        return t_layers, dg_l, g_l, err
                
    
    @staticmethod
    def rf_sc(partial, t0, t1, g0, g1):
        diff = g1 - g0
        numerator = t0 * g1 - t1 * g0
        t = np.linalg.inv(diff) @ numerator
        return t 
    
    @staticmethod
    def conjugate_gradient_pk(pk, gk, gk1, k, cg_formula, restart, last_step):
        if (k == 0) | (restart == True):
            if last_step == 'steepest':
                pk = - gk
                g_k = gk
                g_k1 = gk
            else:
                p_k = - gk
                g_k = gk
                g_k1 = gk1
        else:
            pk = pk
            g_k = gk
            g_k1 = gk1
        if cg_formula == 'Polak':
            pk1 = -gk + (np.linalg.inv(np.matmul(g_k.T, g_k)) @ np.matmul(g_k1.T, g_k1 - g_k)) @ pk
        elif cg_formula == 'Fletcher':
            pk1 = -gk + (np.linalg.inv(np.matmul(g_k.T, g_k)) @ np.matmul(g_k1.T, g_k1)) @ pk
        return pk1, g_k, g_k1
                
    
    def predict(self, X):
        result = []
        N = X.shape[0] # number of observations
        # running through X row by row, so actually the input data is a piece of vector with 1 * D repeated for N times
        for i in range(N):
            # go through all layers in NN
            output = X[i]
            for layer in self.layers:
                output = layer.forward_prop(output)
            result.append(output)
        return result
    