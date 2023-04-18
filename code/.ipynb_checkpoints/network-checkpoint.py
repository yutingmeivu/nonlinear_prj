import numpy as np
import matplotlib.pyplot as plt

class NN:
    def __init__(self, x_train, y_train):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.loss_H = None
        self.x_train = x_train
        self.y_train = y_train
        # self.input_size = input.shape[1]
        # self.output_size = output.shape[1]
        # self.weight = weight if weight else np.random.rand(self.input_size, self.output_size) # D*N matrix
        # self.bias = bias if bias else np.random.rand(1, self.output_size) # 1*N vector
        
    def add(self, layer):
        self.layers.append(layer)
    
    def use(self, loss, loss_prime, loss_H):
        self.loss = loss
        self.loss_prime = loss_prime
        self.loss_H = loss_H
        
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
    
    def initialize(self, E_y, index):
        g_w = np.matmul(self.x_train[index].T, E_y)
        g_b = E_y
        # g_w = None
        # g_b = None
        H_w = None 
        H_b = None
        return g_w, g_b, H_w, H_b

#     def newton_1d(self, t, partial, optim_method, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, j):
#         # t: t_w, t_b
#         # g: g_w, g_b
#         # H: H_w, H_b
#         N = self.x_train.shape[0]
#         output = self.x_train
#         back_error = self.loss_prime(output, self.y_train[j])
#         H_back = self.loss_H(output, self.y_train[j])
#         g_w, g_b, H_w, H_b = self.initialize(back_error)
#         # for index, layer in enumerate(reversed(self.layers)):
#         back_error, g_w, g_b, H_w, H_b = self.back_prop(back_error, H_back, t, None, optim_method, 'newton', True, True, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w, g_b, H_w, H_b, None)
#         t = t - np.matmul(np.linalg.inv(H_w), g_w) if partial == 'w' \
#         else t - np.matmul(np.linalg.inv(H_b), g_b) # since the f' and f'' of newton's method is actually the Jacobian matrix and hessian matrix of objective function, so used matrix manipulation
#         return t
    
#     # def newton_1d(t, J, H):
#     #   return t - J / H
    
#     def generate_startp(self, t_0, rand_low, rand_up, partial, optim_method, K, tolerate, tolerate_g, tolerate_distance, cg_formula):
#         rand = np.random.uniform(rand_low, rand_up)
#         t_1 = t_0 + rand
#         g_w0, g_b0, H_w0, H_b0 = self.initialize(back_error)
#         back_error0, g_w0, g_b0, H_w0, H_b0 = self.layers[-1].back_prop(back_error0, None, t_0, None, optim_method, 'secant', True, True, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w0, g_b0, H_w0, H_b0, None)
#         g_w1, g_b1, H_w1, H_b1 = self.initialize(back_error)
#         back_error1, g_w1, g_b1, H_w1, H_b1 = self.layers[-1].back_prop(back_error1, None, t_1, None, optim_method, 'secant', True, True, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w1, g_b1, H_w1, H_b1, None)
#         det_g0 = np.linalg.det(g_w0) if partial == 'w' else np.linalg.det(g_b0)
#         det_g1 = np.linalg.det(g_w1) if partial == 'w' else np.linalg.det(g_b1)
#         return det_g0, det_g1, back_error0, g_w0, g_b0, t_1, back_error1, g_w1, g_b1
    
#     def regula_falsi(self, t_0, t_1, k, rand_low, rand_up, partial, optim_method, K, tolerate):
#         # n = len(self.layers)
#         back_error0 = self.loss_prime(self.x_train, self.y_train)
#         back_error1 = self.loss_prime(self.x_train, self.y_train)
#         if k == 0:
#             # t_0 = np.zeros(n)
#             t_0 = 0
#             det_g0, det_g1, back_error0, g_w0, g_b0, t_1, back_error1, g_w1, g_b1 = self.generate_startp(t_0, rand_low, rand_up, partial, optim_method, K, tolerate, tolerate_g, tolerate_distance, cg_formula)
#             while np.sign(det_g0 * det_g1) == 1:
#                 det_g0, det_g1, back_error0, g_w0, g_b0, t_1, back_error1, g_w1, g_b1 = self.generate_startp(t_0, rand_low, rand_up, partial, optim_method, K, tolerate, tolerate_g, tolerate_distance, cg_formula)
#         else:
#             # layers_loop = self.layers
#             # for index, layer in enumerate(reversed(layers_loop)):
#             # t_{w+1} = (t_w0 J(w + t_w1 p_w) - t_w1 J(w + t_w0 p_w))/(J(w + t_w1 p_w) - J(w + t_w0 p_w))
#             g_0 = g_w0 if partial == 'w' else g_b0
#             g_1 = g_w1 if partial == 'w' else g_b1
#             diff = g_w1 - g_w0 if partial == 'w' else g_b1 - g_b0
#             t_temp_n = t_0 * g_1 - t_1 * g_0
#             t_tmp = np.linalg.inv(diff) @ t_temp_n
#             back_errort, g_wt, g_bt, H_wt, H_bt = self.back_prop(back_error0, None, t_tmp, None, optim_method, 'regula_falsi', rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w0, g_b0, None, None, None)
#             det_gt = np.linalg.det(g_wt) if partial == 'w' else np.linalg.det(g_bt)
#             if np.sign(det_gt * det_g0) == -1:
#                 t_1 = t_tmp
#                 if partial == 'w':
#                     g_w1 = g_wt 
#                 else:
#                     g_b1 = g_bt
#             else:
#                 t_0 = t_tmp
#                 if partial == 'w':
#                     g_w0 = g_wt
#                 else:
#                     g_b0 = g_bt
#         return t_0, t_1
    
#     def secant_method(self, rand_low, rand_up, partial, optim_method, K, tolerate):
#         rand = np.random.uniform(rand_low, rand_up)
#         # n = len(self.layers)
#         # t_0 = np.zeros(n)
#         t_0 = 0
#         t_1 = t_0 + rand
#         back_error0 = self.loss_prime(self.x_train, self.y_train)
#         back_error1 = back_error0
#         g_w0, g_b0, H_w0, H_b0 = self.initialize(back_error)
#         g_w1, g_b1, H_w1, H_b1 = self.initialize(back_error)
#         # for index, layer in enumerate(reversed(self.layers)):
#         back_error0, g_w0, g_b0, H_w0, H_b0 = self.back_prop(back_error0, None, t_0, None, optim_method, 'secant', True, True, rand_low, rand_up, K, tolerate, None, None, None, g_w0, g_b0, None, None, None)
#         back_error1, g_w1, g_b1, H_w1, H_b1 = self.back_prop(back_error0, None, t_1, None, optim_method, 'secant', True, True, rand_low, rand_up, K, tolerate, None, None, None, g_w1, g_b1, None, None, None)
#         g_0 = g_w0 if partial == 'w' else g_b0
#         g_1 = g_w1 if partial == 'w' else g_b1
#         diff = g_w1 - g_w0 if partial == 'w' else g_b1 - g_b0
#         t_temp_n = t_0 * g_1 - t_1 * g_0
#         t_tmp = np.linalg.inv(diff) @ t_temp_n
#         t_0 = t_1
#         t_1 += t_tmp
#         return t_0, t_1
    
#     def lineSearch(self, t_w, t_b, optim_method, search_method, temp, linesearch, rand_low, rand_up, K, tolerate):
#         # t initial: [0,0,...,0] list with length as same as number of layers
#         # n = len(self.layers)
#         t_w = t_w if t_w else 0 # store in reversed order, which is 0th is the last layer
#         t_b = t_b if t_b else 0
#         k = 0
#         if search_method == 'newton': # for newton and secant method, we don't stop if it diverges, instead just observe the result of loss function, if diverge it would get increases as iteration increases, we can observe its frequency
#             for i in range(4):
#                 t_w = self.newton_1d(t_w, 'w')
#                 t_b = self.newton_1d(t_b, 'b')
#         elif search_method == 'regula_falsi':
#             t_w0, t_w1 = self.regula_falsi(None, None, k_index, rand_low, rand_up, 'w', optim_method, K, tolerate)
#             t_b0, t_b1 = self.regula_falsi(None, None, k_index, rand_low, rand_up, 'b', optim_method, K, tolerate)
#             dis_w = np.linalg.norm(t_w0 - t_w1)
#             dis_b = np.linalg.norm(t_b0 - t_b1)
#             while (k < K) & (dis_w > tolerate) & (dis_b > tolerate):
#                 t_w0, t_w1 = self.regula_falsi(None, None, k_index, rand_low, rand_up, 'w', optim_method, K, tolerate)
#                 t_b0, t_b1 = self.regula_falsi(None, None, k_index, rand_low, rand_up, 'b', optim_method, K, tolerate)
#                 dis_w = np.linalg.norm(t_w0 - t_w1)
#                 dis_b = np.linalg.norm(t_b0 - t_b1)
#                 k += 1
#             t_w = t_w1
#             t_b = t_b1
#         elif search_method == 'secant':
#             t_w0, t_w1 = self.secant_method(None, None, rand_low, rand_up, 'w', optim_method, K, tolerate)
#             t_b0, t_b1 = self.secant_method(None, None, rand_low, rand_up, 'b', optim_method, K, tolerate)
#             dis_w = np.linalg.norm(t_w0 - t_w1)
#             dis_b = np.linalg.norm(t_b0 - t_b1)
#             while (k < K) & (dis_w > tolerate) & (dis_b > tolerate):
#                 t_w0, t_w1 = self.secant_method(None, None, rand_low, rand_up, 'w', optim_method, K, tolerate)
#                 t_b0, t_b1 = self.secant_method(None, None, rand_low, rand_up, 'b', optim_method,  K, tolerate)
#                 k += 1
#             t_w = t_w1
#             t_b = t_b1
#         return t_w, t_b
    
#     def conjugate_gradient_pk(self, pk, gk, gk1, k_index, cg_formula, restart, last_step):
#         if (k_index == 0) | (restart == True):
#             if last_step == 'steepest':
#                 pk = - gk
#                 g_k = gk
#                 g_k1 = gk
#             else:
#                 p_k = - gk
#                 g_k = gk
#                 g_k1 = gk1
#         else:
#             pk = pk
#             g_k = gk
#             g_k1 = gk1
#         if cg_formula == 'Polak':
#             pk1 = -gk + (np.linalg.inv(np.matmul(g_k.T, g_k)) @ np.matmul(g_k1.T, g_k1 - g_k)) @ pk
#         elif cg_formula == 'Fletcher':
#             pk1 = -gk + (np.linalg.inv(np.matmul(g_k.T, g_k)) @ np.matmul(g_k1.T, g_k1)) @ pk
#         return pk1
    
#     def conjugate_gradient(self, partial, search_method, rand_low, rand_up, K, tolerate_g, tolerate_distance, cg_formula, \
#                          g_w0, g_b0, g_w1, g_b1):
#         k = 0
#         step = 0
#         g_w0, g_b0, H_w0, H_b0 = self.initialize()
#         # for index, layer in enumerate(reversed(self.layers)):
#         back_error0, g_w0, g_b0, H_w0, H_b0 = self.back_prop(back_error0, None, None, None, 'steepest', search_method, True, True, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w0, g_b0, H_w0, H_b0, None)
#         g_k = g_w0 if partial == 'w' else g_b0
#         pk1 = self.conjugate_gradient_pk(pk, g_k, None, 0, cg_formula, True, 'steepest')
#         back_error1, g_w1, g_b1, H_w1, H_b1 = self.back_prop(back_error0, None, None, None, 'steepest', search_method, True, True, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w0, g_b0, H_w0, H_b0, pk1)
#         k += 1
#         last = 'conjugate'
#         while (np.linalg.norm(g_w1) > tolerate) & (k < K):
#             n_step = (self.weight.shape[0] * self.weight.shape[1]) - 1 if partial == 'w' else (self.bias.shape[0] * self.bias.shape[1]) - 1
#             dis = np.linalg.norm(np.matmul(np.matmul(g_w0.T, H_w1), g_w1)) if partial == 'w' else np.linalg.norm(np.matmul(np.matmul(g_b0.T, H_b1), g_b1))
#             if (dis <= tolerate_distance) | (step <= n_step) | ():
#                 g_k = g_w0 if partial == 'w' else g_b0
#                 g_k1 = g_w1 if partial == 'w' else g_b1
#                 g_wtmp = g_w1
#                 g_btmp = g_b1
#                 pk1 = pk1 if last == 'conjugate' else - g_k1
#                 pk1 = self.conjugate_gradient_pk(pk1, g_k, g_k1, k, cg_formula, False, last)
#                 back_error1, g_w1, g_b1, H_w1, H_b1 = self.back_prop(back_error0, None, None, None, 'conjugate', search_method, True, True, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w0, g_b0, H_w0, H_b0, pk1)
#                 g_w0 = g_wtmp
#                 g_b0 = g_btmp
#                 k += 1
#                 step += 1
#                 last = 'conjugate'
#             else:
#                 g_k = g_w0 if partial == 'w' else g_b0
#                 g_k1 = g_w1 if partial == 'w' else g_b1
#                 pk1 = - g_k1
#                 pk1 = self.conjugate_gradient_pk(pk1, g_k, g_k1, k, cg_formula, False, 'conjugate')
#                 back_error1, g_w1, g_b1, H_w1, H_b1 = self.back_prop(back_error0, None, None, None, 'conjugate', search_method, True, True, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w0, g_b0, H_w0, H_b0, pk1)
#                 k += 1
#                 step = 0
#                 last = 'steepest'
#         return pk1
    
    def fit(self, x_train, y_train, epochs, eta, optim_method, search_method, temp, linesearch, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula):
        N = x_train.shape[0]
        err = 0
        err_path = []
        for i in range(epochs):
            for j in range(N):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_prop(output)
                err += self.loss(output, y_train[j])
                
                back_error = self.loss_prime(output, y_train[j])
                H_back = self.loss_H(output, y_train[j])
                g_w, g_b, H_w, H_b = self.initialize(back_error, j)
                for index, layer in enumerate(reversed(self.layers)):
                    back_error, g_w, g_b, H_w, H_b = layer.back_prop(back_error, H_back, eta, optim_method, search_method, False, False, rand_low, rand_up, K, tolerate, tolerate_g, tolerate_distance, cg_formula, g_w, g_b, H_w, H_b, None, j, index, None)
                    
            err = err / N
            err_path.append(err)
            # print(f'epoch {i} / {epochs} with error = {round(err, 5)}.')
        plt.plot(err_path)
        plt.title(f'Loss with back propagation weight and bias update with {optim_method} using {search_method} line search')
        plt.xlabel('number of iterations')
        plt.show()
#         return err_path
    
#     def plt_loss(self, error_path, optim_method, search_method):
#         plt.plot(error_path)
#         plt.title(f'Loss with back propagation weight and bias update with {optim_method} using {search_method} line search')
#         plt.xlabel('number of iterations')