# Comparision of performance of nonlinear optimization method used in accelerating multi neuron network

we compared two optimization method: conjugate gradient method with classic gradient descent in fully connected multi neuron network. Newton's method, regula falsi and secant method would be covered. 

## Table of Contents

- [Introduction](#introduction)
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

### [losses.py](https://github.com/yutingmeivu/nonlinear_prj/blob/main/code/losses.py)
  - Set up objective function of problem.

### [network.py](https://github.com/yutingmeivu/nonlinear_prj/blob/main/code/network.py)

  - Construction of multi layer neural network. 

  - #### [load]

  - Add training dataset of X(D by N) and Y(D by 1).

  - #### [add]

  - Add fully connected layer.

  - ##### Parameters

    - ###### `layer` (`tuple`)

    - the dimension of layer, once adding it would initialize the weight matirx with the dimension of tuple and the bias vector with dimension of (1, tuple[1]).

    - ###### `type_` (`string`)

    - specify if the layer is an fully connected layer or an activation layer.

  - #### [activate]

  - Add activation layer.

  - #### [use]

  - Specify loss function to use about optimization

  - ##### Parameters

    - ###### `loss` (`function`)

    - loss function

    - ###### `loss_prime` (`function`)

    - first derivative of loss function

    - ###### `loss_H` (`function`)

    - second derivative of loss function

  - #### [predict]

  - Prediction for dataset.

  - ##### Parameters

    - ###### `X` (`numpy array`)

    - dataset for making prediction.

  - #### [fit]

  - Forward and backforward weight update from layer to layer.

  - ##### Parameters

    - ###### `x_train, y_train` (`numpy array`)

    - Training dataset.

    - ###### `epoch` (`int`)

    - number of iterations specified from user for training.

    - ###### `eta` (`float`)

    - learning rate specified from user for gradient descent. When doing line search it's the step size in each iteration.

    - ###### `optim_method` (`string`)

    - optimization method: ```gd```, ```steepest```, ```conjugate```.

    -  ###### `search_method` (`string`)

    - line search parameters: ```newton```, ```regula_falsi```, ```secant```.

    - ###### `temp` (`Bool`)

    - False by default. Used for control if update weight and bias, when doing line search and conjugate gradient method it is set to be True.

    - ###### `rand_low`, `rand_up` (`float`)

    - Parameter set for regula falsi and secant method for initialize two starting points, number for controlling the range of uniform distributed random number.

    - ###### `K` (`int`)

    - Number for specifying the number of iterations of line search and conjugate gradient method.

    - ###### `tolerate`, `tolerate_g`, `tolerate_distance` (`float`)

    - Stopping criteria of meassuring either gradient or distance between two points are less than $\delta,$ then stop running.

    - ###### `cg_formula` (`string`)

    - Formula function specified by user for conjugate gradient method, parameters can be ```Polak```: Polak-Ribi`ere c. g. formula, ```Fletcher```:  Fletcher-Reeves c. g. formula.

  - #### [forward_prop]

  - Forward propagation.

  - #### [back_prop]

  - Backward propagation.

  - #### [newton_1d]

  - Newton's method for one iteration of line search.
  
  - #### [generate_starp]

  - Initializing starting point for regula falsi and secant method, return t.
  
  - #### [regula_falsi]

  - Regula-Falsi method for doing line search to find appropriate step size for one iteration of line search, return $t_0, t_1.$
  
  - #### [secant_method]

  - Secant method method for doing line search to find appropriate step size for one iteration of line search, return $t_0, t_1.$
  
  - #### [lineSearch]

  - function for doing newton's method, regula-falsi, secant method with different stopping criteria, return $t$ for one iteration of weight update.
  
  - #### [conjugate_gradient_pk]

  - function for doing conjugate gradient method to get search direction $p_k$ in one iteration based on information from last iteration.
  
  - #### [conjugate_gradient]

  - function for doing conjugate gradient method for one iteration of doing weight update. Do either steepest descent or conjugate method based on restart criteria for iterations $K$ times, stop and return $p_k$ if meet stopping criteria.

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
