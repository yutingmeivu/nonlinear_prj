# Comparision of performance of nonlinear optimization method used in accelerating multi neuron network

## Table of Contents

- Introduction
- [Usage](#usage)
- [Examples](#examples)

## Introduction
In machine learning, backpropagation is a widely used algorithm for training feedforward neural networks by updating weight to minimize the error. However, since it used gradient descent as optimization algorithm 
to find the minimum, it often takes a long time for converging on an acceptable solution[@RePEc:spr:aistmt:v:11:y:1959:i:1:p:1-16]. The gradient descent algorithm is especially vulnerable to slow convergence where the error surface consists
of long valleys with steep sides and would be sensitive to the changes in certain directions, which is the case when 'ill-conditioned' case is met[@ruder2016overview]. 
 There're some methods designed which can construct searches to let solutions get rid of characteristic of graident scheme and outperformed the back-propagation method, such as conjugate gradient algorithm[@charalambous1992conjugate], the Modified Back-Propagation Method[@vogl1988accelerating] and some techniques based on gradient method[@haji2021comparison]. 

### optimization method:
- steepest descent
- conjugate gradient
### line search method to find step size t:
- newton's method
- regula falsi
- secant method

## Usage
we compared two optimization method: conjugate gradient method with classic gradient descent in fully connected multi neuron network. Newton's method, regula falsi and secant method would be covered. 

### Examples

Create a toy example for classic gradient descent to fit neural netwrok and visualize the path of loss decreases:

```python
import numpy as np
from network import NN
from activations import tanh, tanh_prime
from losses import mse, mse_prime, mse_second

# training data toy example
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = NN(x_train, y_train)
net.add((2, 3), 'fc')
net.add('activation')
net.add((3, 1), 'fc')
net.add('activation')

# train
net.use(mse, mse_prime, mse_second)
net.fit(x_train, y_train, epochs=1000, eta = 0.1, optim_method = 'gd', search_method = None, temp = None, linesearch = None, \
        rand_low = None, rand_up = None, K = None, tolerate = None, tolerate_g = None, tolerate_distance = None, cg_formula = None)
```
        
You can specify the optimization method with ```optim_method ``` and ```search_method ``` to specify.

Example of doing conjugate method with newton's method as line search in optimization:

```python
net.use(mse, mse_prime, mse_second)
net.fit(x_train, y_train, epochs=1000, eta = 0.1, optim_method = 'conjugate', search_method = 'newton', temp = None, linesearch = None, \
        rand_low = None, rand_up = None, K = 500, tolerate = 0.01, tolerate_g = 0.01, tolerate_distance = 0.01, cg_formula = 'Polak')
```
