import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics.pairwise import cosine_similarity

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
        
    def distribution(self, dist_specify):
        self.dist = dist_specify
        
    def add(self, layer, type_):
        if type_ == 'fc':
            self.weight.append(np.random.rand(layer[0], layer[1])) # D*N matrix
            self.bias.append(np.random.rand(1, layer[1])) # 1*N vector
            self.weight_tmp.append(self.weight[-1])
            self.bias_tmp.append(self.bias[-1])
        else:
            # activation layer doesn't have w & b
            self.weight.append([None])
            self.bias.append([None])
            self.weight_tmp.append(self.weight[-1])
            self.bias_tmp.append(self.bias[-1])
        self.layer += 1
    
    def use(self, loss, loss_prime, loss_h):
        self.loss = loss
        self.loss_prime = loss_prime
        self.loss_h = loss_h
        
    def forward_prop(self, input_data, layer_index, linesearch_, search_pk):
        self.input_layer.append(input_data) # X in each layer
        if self.weight[layer_index][0] is not None:
            if (linesearch_ == True) | (search_pk == True):
                w = self.weight_tmp[layer_index]
                b = self.bias_tmp[layer_index]
            else:
                w = self.weight[layer_index]
                b = self.bias[layer_index]
            self.output = np.dot(input_data, w) + b
        else:
            self.output = self.activation(input_data)
        return self.output
    
    def backward_prop(self, output_error, output_hessian, layer_index, step_size, pk, partial, linesearch_, search_pk, optim_method, linesearch_method):
        # print('output_error in bakcward', output_error)
        # output_error = dE/dY
        if (linesearch_) | (search_pk):
            w = self.weight_tmp[layer_index]
        else:
            w = self.weight[layer_index]
        if self.weight[layer_index][0] is not None:
            input_error = np.dot(output_error, w.T) # dE/dX = dE/dY * W.T
            weights_error = np.dot(self.input_layer[layer_index].T, output_error) # dE/dW = X.T dE/dY
            bias_error = output_error # dE/dB = dE/dY
            # print('shape check', 'd^2 E/dY^2.T', output_hessian.T.shape, 'W', self.weight[layer_index].shape)
            if linesearch_method == 'newton':
                input_h = np.dot(np.dot(w.T, output_hessian), self.weight[layer_index]) + np.dot(output_error, output_hessian) 
                # (d^2 E/dY^2)^T * (dY/dX) + (dE/dY) * (d^2 Y/dX^2)
                # print('shape check of weight', 'self.input_layer[layer_index].T', self.input_layer[layer_index].T.shape, 'd^2 E/dY^2', output_hessian.shape)
                weight_h = np.dot(np.dot(self.input_layer[layer_index].T, output_hessian), self.input_layer[layer_index]) # d^2E/dW^2 = X^T d^2E/dY^2 X
                bias_h = output_hessian # d^2E/dB^2 = d^2E/dY^2
            else:
                input_h = None
                weight_h = None
                bias_h = None
        else:
            # print('layer index', layer_index, 'input_layer', self.input_layer[layer_index], 'weight', self.weight[layer_index - 1])
            input_error = self.activation_prime(self.input_layer[layer_index]) * output_error # dE/dX = dE/dY \odot \sigma'(X)
            weights_error = np.dot(self.input_layer[layer_index].T, output_error * self.activation_prime(self.input_layer[layer_index])) # dE/dW = X.T (dE/dY \odot sigma'(X))
            bias_error = output_error * self.activation_prime(self.input_layer[layer_index]) # dE/dB = dE/dY \odot sigma'(X)
            if linesearch_method == 'newton':
                input_h = np.dot(np.dot(self.weight[layer_index - 1].T, output_hessian), self.weight[layer_index - 1]) * self.activation_prime(self.input_layer[layer_index]) + output_error * self.activation_h(self.input_layer[layer_index]) # (d^2 E/ dY^2) * sigma'(X) * W^T + (dE/dY) * sigma''(X)
                sigma_prime = self.activation_prime(self.input_layer[layer_index])
                sigma_h = self.activation_h(self.input_layer[layer_index])
                weight_h = np.dot(self.input_layer[layer_index].T, (output_hessian * np.dot(sigma_prime.T , sigma_prime) + output_error * sigma_h))
                # X.T * (d^2 E/dY^2 \odot sigma'(Z)^2 + dE/dY \odot sigma''(Z))
                # weight_h = np.dot(self.input_layer[layer_index].T, (np.dot(self.activation_prime(self.input_layer[layer_index]) * output_hessian, np.dot(self.weight[layer_index].T, self.weight[layer_index])) + np.dot(output_error, self.activation_h(self.input_layer[layer_index]) * self.weight[layer_index]) )) # X.T * (sigma'(X) * (d^2 E/dY^2) * W.T^2 + (dE/dY) * (sigma''(X) * W.T))
                bias_h = output_hessian * self.activation_prime(self.input_layer[layer_index]) + output_error * self.activation_h(self.input_layer[layer_index]) # d^2E/dY^2 \odot sigma'(XW+ B) + dE/dY sigma''(XW + B)
            else:
                input_h = None
                weight_h = None
                bias_h = None
            # print('shape check activation,' 'input_h.shape', input_h.shape, 'weight_h', weight_h.shape, 'bias_h', bias_h.shape)
        
        if (linesearch_) | (search_pk):
            # step_size: step_size_w or step_size_b, same as pk
            # linesearch: True(don't update weight until get step size t), same as search_pk
            if optim_method == 'steepest':
                pk = - weights_error if partial == 'w' else - bias_error
            if partial == 'w':
                if self.weight_tmp[layer_index][0] is not None:
                    self.weight_tmp[layer_index] += step_size * pk
            else:
                if self.bias_tmp[layer_index][0] is not None:
                    
                    self.bias_tmp[layer_index] += step_size * pk
        else:
            if optim_method == 'gd':
                if self.weight[layer_index][0] is not None:
                    self.weight[layer_index] -= step_size[0] * weights_error
                    self.weight_tmp[layer_index] = self.weight[layer_index]
                if self.bias[layer_index][0] is not None:
                    self.bias[layer_index] -= step_size[1] * bias_error
                    self.bias_tmp[layer_index] = self.bias[layer_index]

            elif optim_method == 'steepest':
                if self.weight[layer_index][0] is not None:
                    self.weight[layer_index] -= step_size[0][layer_index] * weights_error
                    self.weight_tmp[layer_index] = self.weight[layer_index]
                if self.bias[layer_index][0] is not None:
                    self.bias[layer_index] -= step_size[1][layer_index] * bias_error
                    self.bias_tmp[layer_index] = self.bias[layer_index]
            else:
                # step_size: (step_size_w, step_size_b), same as pk
                if self.weight[layer_index][0] is not None:
                    # print('step size', step_size[0][layer_index], 'pk[0]', pk[0], 'self.weight[layer_index]', self.weight[layer_index].shape)
                    self.weight[layer_index] += step_size[0][layer_index] * pk[0][layer_index]
                    self.weight_tmp[layer_index] = self.weight[layer_index]
                if self.bias[layer_index][0] is not None:
                    self.bias[layer_index] += step_size[1][layer_index] * pk[1][layer_index]
                    self.bias_tmp[layer_index] = self.bias[layer_index]
                # self.weight_tmp[layer_index] = self.weight[layer_index]
                # self.bias_tmp[layer_index] = self.bias[layer_index]
        # print(f'input layer with {layer_index} th', self.input_layer[layer_index])
        # print("in backforward weight error", weights_error)
        return input_error, input_h, weights_error, bias_error, weight_h, bias_h
    
    def fit(self, epochs, eta, optim_method, linesearch_method, rand_low, rand_up, K, tolerate, cg_formula, print_rate, plt_rate, distance, tolerate_err):
        samples = len(self.x_train)
        err_path = []
        i = 0
        err = 100
        k = 0
        restart = [False] * self.layer
        while (i < epochs) & (err > tolerate_err):
            err = 0
            for j in range(samples):
                
                output = self.x_train[j]
                for m in range(self.layer):
                    output = self.forward_prop(output, m, False, False)
                
                err += self.loss(self.y_train[j], output)
                output_error = self.loss_prime(self.y_train[j], output)
                output_hessian = self.loss_h(self.y_train[j], output)
                output_t = output_error.copy()
                output_ht = output_hessian
                
                if optim_method != 'gd':
                    output_st, output_hst, weights_erst, bias_erst, weight_hst, bias_hst = self.backward_prop(output_error, output_hessian, self.layer - 1, 0, None, None, True, False, 'gd', None)
                    self.weight_tmp = self.weight
                    self.bias_tmp = self.bias
                    t_w = self.linesearch(linesearch_method, 'steepest', 'w', weights_erst, rand_low, rand_up, K, tolerate, eta, distance)
                    self.weight_tmp = self.weight
                    self.bias_tmp = self.bias
                    t_b = self.linesearch(linesearch_method, 'steepest', 'b', bias_erst, rand_low, rand_up, K, tolerate, eta, distance)

                    self.weight_tmp = self.weight
                    self.bias_tmp = self.bias
                    output_stw = output_t.copy()
                    output_stb = output_t.copy()
                    output_hstw = output_ht
                    output_hstb = output_ht

                    if optim_method == 'conjugate':
                        gw_tmp = []
                        gb_tmp = []
                        for n in reversed(range(self.layer)):
                            output_stw, output_hstw, weights_erst, bias_erst, weight_hst, bias_hst = self.backward_prop(output_stw, output_hstw, n, t_w[n], None, None, True, False, 'steepest', linesearch_method)
                            
                            gw_tmp.append(weights_erst)
                            self.weight_tmp = self.weight
                            self.bias_tmp = self.bias
                            output_stb, output_hstb, weights_erst, bias_erst, weight_hst, bias_hst = self.backward_prop(output_stb, output_hstb, n, t_b[n], None, None, True, False, 'steepest', linesearch_method)
                            gb_tmp.append(bias_erst)
                            self.weight_tmp = self.weight
                            self.bias_tmp = self.bias

                        self.weight_tmp = self.weight
                        self.bias_tmp = self.bias
                        gw = [i for i in reversed(gw_tmp)]
                        gb = [i for i in reversed(gb_tmp)]
                        if k == 0:
                            gw1 = gw.copy()
                            pk_w = None
                            gb1 = gb.copy()
                            pk_b = None
                        
                        
                        pk_w, gw, gw1, restart = self.conjugate_gradient(linesearch_method, 'w', t_w, pk_w, gw, gw1, rand_low, rand_up, K, tolerate, cg_formula, restart, k)
                        self.weight_tmp = self.weight
                        self.bias_tmp = self.bias
                        pk_b, gb, gb1, restart = self.conjugate_gradient(linesearch_method, 'b', t_b, pk_b, gb, gb1, rand_low, rand_up, K, tolerate, cg_formula, restart, k)
                        self.weight_tmp = self.weight
                        self.bias_tmp = self.bias

                for n in reversed(range(self.layer)):
                    if optim_method == 'gd':
                        output_error, output_hessian, weights_error, bias_error, weight_h, bias_h = self.backward_prop(output_error, output_hessian, n, (eta, eta), None, None, False, False, optim_method, linesearch_method)
                    elif optim_method == 'steepest':
                        output_error, output_hessian, weights_error, bias_error, weight_h, bias_h = self.backward_prop(output_error, output_hessian, n, (t_w, t_b), None, None, False, False, optim_method, linesearch_method)
                        
                    elif optim_method == 'conjugate':
                        output_error, output_hessian, weights_error, bias_error, weight_h, bias_h = self.backward_prop(output_error, output_hessian, n, (t_w, t_b), (pk_w, pk_b), None, False, False, optim_method, linesearch_method)
            i += 1
            k += 1
                
            err /= samples
            err_path.append(err)
        if print_rate:
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
        if plt_rate:
            plt.plot(err_path)
            plt.title(f'Loss with back propagation weight and bias update with {optim_method} using {linesearch_method} line search')
            plt.xlabel('number of iterations')
            plt.show()
        print(f'final error reaches at {err} with {i} iterations.')
    
    @staticmethod
    def newton1d(t, partial, g, h): 
        t = t - np.matmul(np.linalg.inv(h), g) # since the f' and f'' of newton's method is actually the Jacobian matrix and hessian matrix of objective function, so used matrix manipulation
        return t 
    
    @staticmethod
    def trans(g):
        if g.ndim == 1:
            g = g.reshape(-1,1)
        return g
            
    def linesearch(self, linesearch_method, optim_method, partial, start_g, rand_low, rand_up, K, tolerate, eta, distance):
        # start_g: gradient of either weight and bias from fit
        if linesearch_method == 'newton':
            t_layers = [0] * len(self.weight)
            dgtmp = [0] * len(self.weight)
            gtmp = [0] * len(self.weight)
            err = [0] * len(self.weight)
            for i in range(4):
                t_layers = self.subupdatesubupdate(K, t_layers, None, start_g, partial, linesearch_method, optim_method)
                self.weight_tmp = self.weight
                self.bias_tmp = self.bias
        else:
            # rand = np.random.uniform(rand_low, rand_up)
            t_layers0 = [0] * len(self.weight)
            rand = self.dist(rand_low, rand_up)
            t_layers1 = [rand] * len(self.weight)
            t_tmp = [None] * len(self.weight)
            k = 0
            dg0 = [None] * len(self.weight)
            dg1 = [None] * len(self.weight)
            if linesearch_method == 'regula_falsi':
                rand = self.dist(rand_low, rand_up)
                t_layers0 = [rand] * len(self.weight)
                dg0 = self.generate_startp(t_layers0, partial, k, start_g, optim_method, linesearch_method, dg0)
                # print('self.weight_tmp generate p dg0', self.weight_tmp)
                self.weight_tmp = self.weight
                self.bias_tmp = self.bias
                dg1 = self.generate_startp(t_layers1, partial, k, start_g, optim_method, linesearch_method, dg1)
                # print('self.weight_tmp generate p dg1', self.weight_tmp)
                self.weight_tmp = self.weight
                self.bias_tmp = self.bias
                # print('dg0', dg0)
                # print('dg1', dg1)
                for i in range(len(t_layers0)):
                    print('cos', cosine_similarity(self.trans(dg1[i]), self.trans(dg0[i])))
                    print('dg0[i]', dg0[i])
                    print('dg1[i]', dg1[i])
                    while cosine_similarity(self.trans(dg1[i]), self.trans(dg0[i])) > 0:
                        print('while loop')
                        print('cos', cosine_similarity(self.trans(dg1[i]), self.trans(dg0[i])))
                        print('dg0[i]', dg0[i])
                        print('dg1[i]', dg1[i])
                        # print('dg1', dg1, 'dg0', dg0)
                        rand = self.dist(rand_low, rand_up)
                        print('rand', rand)
                        t_layers1[i] = rand
                        dg1 = self.generate_startp(t_layers1, partial, k, start_g, optim_method, linesearch_method, dg1)
                        self.weight_tmp = self.weight
                        self.bias_tmp = self.bias
                    print('while loop end')

            g0 = start_g.copy()
            g1 = start_g.copy()
            gt = start_g.copy()
            # print(t_layers0, t_layers1)
            dis = np.linalg.norm(list(map(lambda x,y: x - y, t_layers0, t_layers1)))
            if np.isnan(g0):
                g0 = np.array([np.random.rand()]).reshape(-1,1)
            if np.isnan(g1):
                g1 = np.array([np.random.rand()]).reshape(-1,1)
            cosd = np.abs(cosine_similarity(np.asarray(g0).reshape(-1,1), np.asarray(g1).reshape(-1,1)))
            # cosine_similarity(np.array(g1), np.array(g0))
            #  & (np.linalg.norm(g1 - g0) > tolerate)
            # & (np.mean(list(map(lambda x: np.linalg.norm(x), g1))) > tolerate) & (np.mean(list(map(lambda x: np.linalg.norm(x), g0))) > tolerate)
            while (k < K) & (dis > tolerate) & (cosd > tolerate)\
            & (np.mean(list(map(lambda x: np.linalg.norm(x), g1))) > tolerate) & (np.mean(list(map(lambda x: np.linalg.norm(x), g0))) > tolerate):
                tmp, dg0, g0, err0 = self.subupdate(k, t_layers0, None, g0, partial, linesearch_method, optim_method)
                self.weight_tmp = self.weight
                self.bias_tmp = self.bias
                tmp, dg1, g1, err1 = self.subupdate(k, t_layers1, None, g1, partial, linesearch_method, optim_method)
                self.weight_tmp = self.weight
                self.bias_tmp = self.bias
                for i in range(len(dg0)):
                    
                    t_tmp[i] = self.rf_sc(partial, t_layers0[i], t_layers1[i], g0[i], g1[i], eta, distance)
                tmp, dgt, gt, errt = self.subupdate(i, tmp, None, gt, partial, linesearch_method, optim_method)
                self.weight_tmp = self.weight
                self.bias_tmp = self.bias
                if linesearch_method == 'regula_falsi':
                    for i in range(len(t_layers[0])):
                        # np.dot(g0[i].flatten(), gt[i].flatten())
                        
                        if cosine_similarity(self.trans(g0[i]), self.trans(gt[i])) < 0:
                        # if np.norm(g0[i]) * np.norm(gt[i]) < 0:
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
                dis = math.sqrt(sum(list(map(lambda x,y: (x - y)**2, t_layers0, t_layers1))))
                k += 1

                g0 = list(map(lambda x: self.trans(x), g0))
                g1 = list(map(lambda x: self.trans(x), g1))

                cosd = np.abs(np.mean(list(map(lambda x,y: np.mean(cosine_similarity(x, y)), g0, g1))))
                
            self.weight_tmp = self.weight
            self.bias_tmp = self.bias
            if linesearch_method == 'regula_falsi':
                t_layers = t_layers0 if err0 < err1 else t_layers1
            else:
                t_layers = t_layers1
        return t_layers # t_layers has forward order of moving step in each layers
    
    def conjugate_gradient(self, linesearch_method, partial, t, pk, gk, gk1, rand_low, rand_up, K, tolerate, cg_formula, restart, k):
        # k = 0
        # restart = [False] * len(self.weight)
        n_step = [0] * len(self.weight)
        gk1 = gk.copy() if k == 0 else gk1
        restart = self.n_step_check(restart, k, partial)
        pk = gk.copy() if k == 0 else pk
        prev_step = ['steepest'] * len(self.weight_tmp)
        # while (np.mean(list(map(lambda x: np.linalg.norm(x), gk1))) > tolerate) & (k < K):
            # t = self.linesearch(self, linesearch_method, optim_method, partial, gk1, rand_low, rand_up, K, tolerate)
        for i in reversed(range(self.layer)):
            if (k == 0) | (restart[i] == True):
                gk = gk1
                t, dg_l, gk1, err = self.subupdate(k, t, pk, gk1, partial, linesearch_method, 'steepest')
                self.weight_tmp = self.weight
                self.bias_tmp = self.bias
                pk[i] = -gk[i].copy()
                prev_step[i] = 'steepest'
            else:
                pk[i], gk[i], gk1[i] = self.conjugate_gradient_pk(pk[i], gk[i], gk1[i], k, cg_formula, restart[i], prev_step[i])
                prev_step[i] = 'conjugate'
                t, dg_l, gk1, err = self.subupdate(k, t, pk, gk1, partial, linesearch_method, 'conjugate')
                self.weight_tmp = self.weight
                self.bias_tmp = self.bias

        restart = self.n_step_check(restart, k, partial)
        return pk, gk, gk1, restart
            
    
    def n_step_check(self, restart, k, partial):
        n_step = [None] * self.layer
        for i in range(self.layer):
            if partial == 'w':
                if self.weight_tmp[i][0] is not None:
                    param = self.weight_tmp[i]
            else:
                if self.bias_tmp[i][0] is not None:
                    param = self.bias_tmp[i]
                # n_step[i] = (param.shape[0] * param.shape[1]) - 1
            if param.ndim == (1,):
                n_step[i] = len(param)
            else:
                n_step[i] = param.shape[0]
            restart[i] = True if k >= n_step[i] else False
        return restart
    
    def generate_startp(self, t1, partial, k, start_g, optim_method, linesearch_method, detg_l):
        samples = len(self.x_train)
        
        for i in range(2):
            err = 0
            for j in range(samples):
                output = self.x_train[j]
                for m in range(self.layer):
                    output = self.forward_prop(output, m, True, False)
                        
                err += self.loss(self.y_train[j], output)
                output_error = self.loss_prime(self.y_train[j], output)
                output_hessian = self.loss_h(self.y_train[j], output)
                
                for n in reversed(range(self.layer)):
                    output_error, output_hessian, weights_error, bias_error, weight_h, bias_h = self.backward_prop(output_error, output_hessian, n, t1[n], None, partial, True, False, optim_method, linesearch_method)
                    detg_l[n] = weights_error if partial == 'w' else bias_error
                    if partial == 'w':
                        if self.weight_tmp[n][0] is not None:
                            self.weight_tmp[n] -= t1[n] * weights_error
                    else:
                        if self.bias_tmp[n][0] is not None:
                            self.bias_tmp[n] -= t1[n] * bias_error      
        return detg_l
    
    def subupdate(self, k, t_layers, pk, start_g, partial, linesearch_method, optim_method):
        samples = len(self.x_train)
        err = 0
        dg_l = [None] * len(self.weight)
        g_l = [None] * len(self.weight)
        for j in range(samples):
            output = self.x_train[j]
            for m in range(self.layer):
                output = self.forward_prop(output, m, True, False)
                            
            err += self.loss(self.y_train[j], output)
            output_error = self.loss_prime(self.y_train[j], output)
            output_hessian = self.loss_h(self.y_train[j], output)
            if optim_method != 'conjugate':
                for n in reversed(range(self.layer)):
                    output_error, output_hessian, weights_error, bias_error, weight_h, bias_h = self.backward_prop(output_error, output_hessian, n, t_layers[n], None, partial, True, False, optim_method, linesearch_method)
                    if linesearch_method == 'newton':
                        h = weight_h if partial == 'w' else bias_h
                        g = weights_error if partial == 'w' else bias_error
                        t_layers[n] = self.newton1d(t_layers[n], partial, g, h)
                        g_l[n] = weights_error if partial == 'w' else bias_error
                    else:
                            # dg_l[n] = np.linalg.det(weights_error) if partial == 'w' else np.linalg.det(bias_error)
                        g_l[n] = weights_error if partial == 'w' else bias_error
                    if k == 0:
                        weights_error = start_g
                        bias_error = start_g
                    if partial == 'w':
                        if self.weight_tmp[m][0] is not None:
                            self.weight_tmp[m] -= t_layers[m] * weights_error
                    else:
                        if self.bias_tmp[m][0] is not None:
                            self.bias_tmp[m] -= t_layers[m] * bias_error
            else:
                for n in reversed(range(self.layer)):
                    output_error, output_hessian, weights_error, bias_error, weight_h, bias_h = self.backward_prop(output_error, output_hessian, n, t_layers[n], pk[n], partial, False, True, optim_method, linesearch_method)

                    if partial == 'w':
                        g_l[n] = weights_error
                    else:
                        g_l[n] = bias_error
        return t_layers, dg_l, g_l, err

    
    def rf_sc(self, partial, t0, t1, g0, g1, eta, distance):
        # cosine similarity
        if distance == 'cosine':
            diff = np.mean(cosine_similarity(self.trans(g1), self.trans(g0)))
            numerator = np.mean(cosine_similarity(t0 * self.trans(g1), t1 * self.trans(g0)))
            t = (numerator / diff) if diff != 0 else 1
        
        # diff = np.linalg.norm(g1 - g0)
        # numerator = np.linalg.norm(t0 * g1 - t1 * g0)

        # t = np.dot(np.linalg.pinv(diff), numerator)
        # t = np.linalg.pinv(diff) @ numerator
        # diff = diff if diff != 0 else 1.0

        # Frobenius norm
        elif distance == 'fro':
            diff = np.linalg.norm(g1 - g0, ord='fro')
            numerator = np.linalg.norm(t0 * self.trans(g1) - t1 * self.trans(g0), ord = 'fro')
            t = (numerator / diff) if diff != 0 else 1
        
        return t 
    
    def conjugate_gradient_pk(self, pk, gk, gk1, k, cg_formula, restart, last_step):
        if (k == 0) | (restart == True):
            # if last_step == 'steepest':
            pk = - gk
            g_k = gk
            g_k1 = gk1
        else:
            pk = pk
            g_k = gk
            g_k1 = gk1
        if cg_formula == 'Polak':
            # pk1 = -gk + (np.linalg.inv(np.matmul(g_k.T, g_k)) @ np.matmul(g_k1.T, g_k1 - g_k)) @ pk
            dis = np.mean(cosine_similarity(self.trans(np.dot(g_k.T, g_k1)), self.trans(np.dot(g_k.T, g_k))))
            numerator = dis if not np.isnan(dis) else 0
            denominator = np.linalg.norm(g_k, ord=2)**2
        elif cg_formula == 'Fletcher':
            numerator = np.linalg.norm(g_k1, ord=2)**2
            denominator = np.linalg.norm(g_k, ord=2)**2
        pk1 = - gk + (numerator / denominator) * pk

        return pk1, g_k, g_k1
                
    
    def predict(self, X):
        result = []
        N = X.shape[0] # number of observations
        # running through X row by row, so actually the input data is a piece of vector with 1 * D repeated for N times
        for i in range(N):
            # go through all layers in NN
            output = X[i]
            for layer in self.layers:
                output = layer.forward_prop(output, False, False)
            result.append(output)
        return result
    