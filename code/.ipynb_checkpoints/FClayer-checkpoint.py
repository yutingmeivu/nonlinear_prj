import numpy as np 
from layer import Layer

class fcLayer(Layer):
    def __init__(self, input, output, x_train, y_train, loss, loss_prime, loss_H, weight = None, bias = None) -> None:
        # self.input = None # row slice of D*N matrix
        # self.output = None # 1*N vector
        self.input_size = input
        self.output_size = output
        self.weight = weight if weight else np.random.rand(self.input_size, self.output_size) # D*N matrix
        self.bias = bias if bias else np.random.rand(1, self.output_size) # 1*N vector
        self.loss = loss
        self.loss_prime = loss_prime
        self.loss_H = loss_H
        self.x_train = x_train
        self.y_train = y_train
        self.weight_tmp = self.weight.copy()
        self.bias_tmp = self.bias.copy()
        self.line = False
    
    def forward_prop(self, input_data):
        self.input = input_data
        w = self.weight if self.line == False else self.weight_tmp
        b = self.bias if self.line == False else self.weight_tmp
        self.output = np.matmul(self.input, w) + b
        return self.output
    
    def back_prop(self, E_y, E_y2, eta, optim_method, search_method, temp, linesearch, rand_low, rand_up, K, tolerate, \
                tolerate_g, tolerate_distance, cg_formula, g_w = None, g_b = None, H_w = None, H_b = None, pk = None, j = None, index = None, partial = None, l = None):
        # E_y output_err: error comes from the layer behid the current layer (dE/ dY)
        # update items: weight, bias
        # output: error from current layer with (w,b), supposed to pass into the previous layer
        # temp: if True, then it's doing line search something to find t candidate, we don't want to update w and b
        # print('backward dE dY', E_y.shape, 'self.weight.T', self.weight.T.shape, 'self.input.T, X.T', self.input.T.shape)
        # print('*'*10)
        g_x = np.matmul(E_y, self.weight.T) # dE/dX = dE/dY times W.T
        g_w = np.matmul(self.input.T, E_y)
        g_b = E_y
        if ((optim_method == 'steepest') | (optim_method == 'conjugate')) & (search_method == 'newton'):    
            H_w = E_y2 * np.matmul(self.input.T, self.input)
            H_b = E_y2
        else:
            H_w = None
            H_b = None
        if optim_method == 'gd':
            dk_weight = - eta * g_w
            dk_bias = - eta * E_y
        else:
            if optim_method == 'steepest':
                pkw = -g_w
                pkb = -g_b
                t_w, t_b = self.lineSearch(None, None, optim_method, search_method, temp, linesearch, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, j, E_y, E_y2, g_w, g_b, H_w, H_b, pk)
                dk_weight = t_w[j][index] * pkw
                dk_bias = t_b[j][index] * pkb
                self.line = True
            elif optim_method == 'conjugate':
                t_w, t_b = self.lineSearch(None, None, optim_method, search_method, temp, linesearch, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, j, E_y, E_y2, g_w, g_b, H_w, H_b, pk)
                pkw = self.conjugate_gradient('w', search_method, rand_low, rand_up, K, tolerate_g, tolerate_distance, cg_formula, \
                                 None, None, None, None)
                pkb = self.conjugate_gradient('b', search_method, rand_low, rand_up, K, tolerate_g, tolerate_distance, cg_formula, \
                                 None, None, None, None)
                dk_weight = t_w[j][index] * pkw
                dk_bias = t_b[j][index] * pkb
        self.weight_tmp += dk_weight
        self.bias_tmp += dk_bias
        if (temp == False) | (optim_method == 'gd'):
            self.weight += dk_weight
            self.bias += dk_bias
            self.weight_tmp = self.weight.copy()
            self.bias = self.bias.copy()
            self.line = True
            # self.weight_layer[self.layer_num - 1] = self.weight 
            # self.bias_layer[self.layer_num - 1] = self.bias
        return g_x, g_w, g_b, H_w, H_b
    
    def initialize(self, E_y, index):
        g_w = np.matmul(self.x_train[index].T, E_y)
        g_b = E_y
        # g_w = None
        # g_b = None
        H_w = None 
        H_b = None
        return g_w, g_b, H_w, H_b    
    
    @staticmethod
    def newton_1d(t, partial, E_y, E_y2, g_w, g_b, H_w, H_b):
        # t: t_w, t_b
        # g: g_w, g_b
        # H: H_w, H_b
        # N = self.input.shape[0]
#         output = self.input
        
#         back_error = self.loss_prime(output, self.y_train[j])
#         H_back = self.loss_H(output, self.y_train[j])
#         g_w, g_b, H_w, H_b = self.initialize(back_error, j)
#         # for index, layer in enumerate(reversed(self.layers)):
#         print('newton before', 'newton dE/dY', back_error.shape, 'newton dE^2/dY^2, number', 'newton before dE/dW', g_w.shape)
#         print("="*10)
#         back_error, g_w, g_b, H_w, H_b = self.back_prop(back_error, H_back, t, optim_method, 'newton', True, True, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w, g_b, H_w, H_b, None, j)
#         print('newton after', 'newton dE/dY', back_error.shape, 'newton dE^2/dY^2 number', 'newton after dE/dW', g_w.shape)
        t = t - np.matmul(np.linalg.inv(H_w), g_w) if partial == 'w' \
        else t - np.matmul(np.linalg.inv(H_b), g_b) # since the f' and f'' of newton's method is actually the Jacobian matrix and hessian matrix of objective function, so used matrix manipulation
        return t
    
    # def newton_1d(t, J, H):
    #   return t - J / H
    
    def generate_startp(self, t_0, rand_low, rand_up, partial, optim_method, K, tolerate, tolerate_g, tolerate_distance, cg_formula, j):
        rand = np.random.uniform(rand_low, rand_up)
        t_1 = t_0 + rand
        g_w0, g_b0, H_w0, H_b0 = self.initialize(back_error)
        back_error0, g_w0, g_b0, H_w0, H_b0 = self.layers[-1].back_prop(back_error0, None, t_0, None, optim_method, 'secant', True, True, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w0, g_b0, H_w0, H_b0, None)
        g_w1, g_b1, H_w1, H_b1 = self.initialize(back_error)
        back_error1, g_w1, g_b1, H_w1, H_b1 = self.layers[-1].back_prop(back_error1, None, t_1, None, optim_method, 'secant', True, True, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w1, g_b1, H_w1, H_b1, None)
        det_g0 = np.linalg.det(g_w0) if partial == 'w' else np.linalg.det(g_b0)
        det_g1 = np.linalg.det(g_w1) if partial == 'w' else np.linalg.det(g_b1)
        return det_g0, det_g1, back_error0, g_w0, g_b0, t_1, back_error1, g_w1, g_b1
    
    def regula_falsi(self, t_0, t_1, k, rand_low, rand_up, partial, optim_method, K, tolerate, j):
        # n = len(self.layers)
        back_error0 = self.loss_prime(self.x_train, self.y_train)
        back_error1 = self.loss_prime(self.x_train, self.y_train)
        if k == 0:
            # t_0 = np.zeros(n)
            t_0 = 0
            det_g0, det_g1, back_error0, g_w0, g_b0, t_1, back_error1, g_w1, g_b1 = self.generate_startp(t_0, rand_low, rand_up, partial, optim_method, K, tolerate, tolerate_g, tolerate_distance, cg_formula, j)
            while np.sign(det_g0 * det_g1) == 1:
                det_g0, det_g1, back_error0, g_w0, g_b0, t_1, back_error1, g_w1, g_b1 = self.generate_startp(t_0, rand_low, rand_up, partial, optim_method, K, tolerate, tolerate_g, tolerate_distance, cg_formula, j)
        else:
            # layers_loop = self.layers
            # for index, layer in enumerate(reversed(layers_loop)):
            # t_{w+1} = (t_w0 J(w + t_w1 p_w) - t_w1 J(w + t_w0 p_w))/(J(w + t_w1 p_w) - J(w + t_w0 p_w))
            g_0 = g_w0 if partial == 'w' else g_b0
            g_1 = g_w1 if partial == 'w' else g_b1
            diff = g_w1 - g_w0 if partial == 'w' else g_b1 - g_b0
            t_temp_n = t_0 * g_1 - t_1 * g_0
            t_tmp = np.linalg.inv(diff) @ t_temp_n
            back_errort, g_wt, g_bt, H_wt, H_bt = self.back_prop(back_error0, None, t_tmp, None, optim_method, 'regula_falsi', rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w0, g_b0, None, None, None, j)
            det_gt = np.linalg.det(g_wt) if partial == 'w' else np.linalg.det(g_bt)
            if np.sign(det_gt * det_g0) == -1:
                t_1 = t_tmp
                if partial == 'w':
                    g_w1 = g_wt 
                else:
                    g_b1 = g_bt
            else:
                t_0 = t_tmp
                if partial == 'w':
                    g_w0 = g_wt
                else:
                    g_b0 = g_bt
        return t_0, t_1
    
    def secant_method(self, rand_low, rand_up, partial, optim_method, K, tolerate, j):
        rand = np.random.uniform(rand_low, rand_up, j)
        # n = len(self.layers)
        # t_0 = np.zeros(n)
        t_0 = 0
        t_1 = t_0 + rand
        back_error0 = self.loss_prime(self.x_train, self.y_train)
        back_error1 = back_error0
        g_w0, g_b0, H_w0, H_b0 = self.initialize(back_error)
        g_w1, g_b1, H_w1, H_b1 = self.initialize(back_error)
        # for index, layer in enumerate(reversed(self.layers)):
        back_error0, g_w0, g_b0, H_w0, H_b0 = self.back_prop(back_error0, None, t_0, None, optim_method, 'secant', True, True, rand_low, rand_up, K, tolerate, None, None, None, g_w0, g_b0, None, None, None)
        back_error1, g_w1, g_b1, H_w1, H_b1 = self.back_prop(back_error0, None, t_1, None, optim_method, 'secant', True, True, rand_low, rand_up, K, tolerate, None, None, None, g_w1, g_b1, None, None, None)
        g_0 = g_w0 if partial == 'w' else g_b0
        g_1 = g_w1 if partial == 'w' else g_b1
        diff = g_w1 - g_w0 if partial == 'w' else g_b1 - g_b0
        t_temp_n = t_0 * g_1 - t_1 * g_0
        t_tmp = np.linalg.inv(diff) @ t_temp_n
        t_0 = t_1
        t_1 += t_tmp
        return t_0, t_1
    
    def lineSearch(self, t_w, t_b, optim_method, search_method, temp, linesearch, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, j, E_y, E_y2, g_w, g_b, H_w, H_b, pk):
        # t initial: [0,0,...,0] list with length as same as number of layers
        # n = len(self.layers)
        t_w = t_w if t_w else 0 # store in reversed order, which is 0th is the last layer
        t_b = t_b if t_b else 0
        k = 0
        N = self.x_train.shape[0]
        t_ws = [t_w]
        t_bs = [t_b]
        t_wsum = []
        t_bsum = []
        # for newton and secant method, we don't stop if it diverges, instead just observe the result of loss function, if diverge it would get increases as iteration increases, we can observe its frequency
        
        for i in range(4):
            for j in range(N):
                output = self.x_train[j]
                for layer in self.layers:
                    output = layer.forward_prop(output)
                err += self.loss(output, y_train[j])
                    
                back_error = self.loss_prime(output, y_train[j])
                H_back = self.loss_H(output, y_train[j])
                g_w, g_b, H_w, H_b = self.initialize(back_error, j)
                
                for index, layer in enumerate(reversed(self.layers)):
                    if search_method == 'newton':
                        back_error, g_w, g_b, H_w, H_b = layer.back_prop(back_error, H_back, t_w, optim_method, search_method, False, False, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w, g_b, H_w, H_b, None, j, index, 'w')
                        
                        t_ws.append(newton_1d(t_ws[j][index], 'w', backerror, H_back, g_w, g_b, H_w, H_b))
                        back_error, g_w, g_b, H_w, H_b = layer.back_prop(back_error, H_back, t_b, optim_method, search_method, False, False, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w, g_b, H_w, H_b, None, j, index,'b')
                        t_bs.append(newton_1d(t_bs[j][index], 'b', backerror, H_back, g_w, g_b, H_w, H_b))
                        
                    elif search_method == 'regula_falsi':
                        t_w0, t_w1 = self.regula_falsi(None, None, k_index, rand_low, rand_up, 'w', optim_method, K, tolerate, j)
                        t_b0, t_b1 = self.regula_falsi(None, None, k_index, rand_low, rand_up, 'b', optim_method, K, tolerate, j)
                        dis_w = np.linalg.norm(t_w0 - t_w1)
                        dis_b = np.linalg.norm(t_b0 - t_b1)
                        while (k < K) & (dis_w > tolerate) & (dis_b > tolerate):
                            t_w0, t_w1 = self.regula_falsi(None, None, k_index, rand_low, rand_up, 'w', optim_method, K, tolerate, j)
                            t_b0, t_b1 = self.regula_falsi(None, None, k_index, rand_low, rand_up, 'b', optim_method, K, tolerate, j)
                            dis_w = np.linalg.norm(t_w0 - t_w1)
                            dis_b = np.linalg.norm(t_b0 - t_b1)
                            k += 1
                        t_w = t_w1
                        t_b = t_b1
                    elif search_method == 'secant':
                        t_w0, t_w1 = self.secant_method(None, None, rand_low, rand_up, 'w', optim_method, K, tolerate, j)
                        t_b0, t_b1 = self.secant_method(None, None, rand_low, rand_up, 'b', optim_method, K, tolerate, j)
                        dis_w = np.linalg.norm(t_w0 - t_w1)
                        dis_b = np.linalg.norm(t_b0 - t_b1)
                        while (k < K) & (dis_w > tolerate) & (dis_b > tolerate):
                            t_w0, t_w1 = self.secant_method(None, None, rand_low, rand_up, 'w', optim_method, K, tolerate, j)
                            t_b0, t_b1 = self.secant_method(None, None, rand_low, rand_up, 'b', optim_method,  K, tolerate, j)
                            k += 1
                        t_w = t_w1
                        t_b = t_b1
                t_wsum.append(t_ws)
                t_bsum.append(t_bs)
        return t_wsum, t_bsum
    
    def conjugate_gradient_pk(self, pk, gk, gk1, k_index, cg_formula, restart, last_step):
        if (k_index == 0) | (restart == True):
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
        return pk1
    
    def conjugate_gradient(self, partial, search_method, rand_low, rand_up, K, tolerate_g, tolerate_distance, cg_formula, \
                         g_w0, g_b0, g_w1, g_b1):
        k = 0
        step = 0
        g_w0, g_b0, H_w0, H_b0 = self.initialize()
        # for index, layer in enumerate(reversed(self.layers)):
        back_error0, g_w0, g_b0, H_w0, H_b0 = self.back_prop(back_error0, None, None, None, 'steepest', search_method, True, True, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w0, g_b0, H_w0, H_b0, None)
        g_k = g_w0 if partial == 'w' else g_b0
        pk1 = self.conjugate_gradient_pk(pk, g_k, None, 0, cg_formula, True, 'steepest')
        back_error1, g_w1, g_b1, H_w1, H_b1 = self.back_prop(back_error0, None, None, None, 'steepest', search_method, True, True, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w0, g_b0, H_w0, H_b0, pk1)
        k += 1
        last = 'conjugate'
        while (np.linalg.norm(g_w1) > tolerate) & (k < K):
            n_step = (self.weight.shape[0] * self.weight.shape[1]) - 1 if partial == 'w' else (self.bias.shape[0] * self.bias.shape[1]) - 1
            dis = np.linalg.norm(np.matmul(np.matmul(g_w0.T, H_w1), g_w1)) if partial == 'w' else np.linalg.norm(np.matmul(np.matmul(g_b0.T, H_b1), g_b1))
            if (dis <= tolerate_distance) | (step <= n_step) | ():
                g_k = g_w0 if partial == 'w' else g_b0
                g_k1 = g_w1 if partial == 'w' else g_b1
                g_wtmp = g_w1
                g_btmp = g_b1
                pk1 = pk1 if last == 'conjugate' else - g_k1
                pk1 = self.conjugate_gradient_pk(pk1, g_k, g_k1, k, cg_formula, False, last)
                back_error1, g_w1, g_b1, H_w1, H_b1 = self.back_prop(back_error0, None, None, None, 'conjugate', search_method, True, True, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w0, g_b0, H_w0, H_b0, pk1)
                g_w0 = g_wtmp
                g_b0 = g_btmp
                k += 1
                step += 1
                last = 'conjugate'
            else:
                g_k = g_w0 if partial == 'w' else g_b0
                g_k1 = g_w1 if partial == 'w' else g_b1
                pk1 = - g_k1
                pk1 = self.conjugate_gradient_pk(pk1, g_k, g_k1, k, cg_formula, False, 'conjugate')
                back_error1, g_w1, g_b1, H_w1, H_b1 = self.back_prop(back_error0, None, None, None, 'conjugate', search_method, True, True, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w0, g_b0, H_w0, H_b0, pk1)
                k += 1
                step = 0
                last = 'steepest'
        return pk1
    
    
    