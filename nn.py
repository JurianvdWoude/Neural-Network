import numpy as np

def initialize_parameters(n_x, n_h, n_y):
  W1 = np.random.randn(n_h, n_x) * 0.01
  b1 = np.zeros((n_h, 1))
  W2 = np.random.randn(n_y, n_h) * 0.01
  b2 = np.zeros((n_y, 1))
  parameters = {
    "W1": W1,
    "b1": b1,
    "W2": W2,
    "b2": b2
  }

  return parameters


# --------------------------------------------------------------

def linear_forward(A, W, b):
  Z = np.dot(W, A) + b
  cache = (A, W, b)
  return Z, cache

def sigmoid(Z):
  A = 1/(1 + np.exp(-Z))
  return A, Z

def relu(Z):
  A = max(0, Z)
  return A, Z

def initialize_parameters_deep(layer_dims):
  """
  Arguments:
  layer_dims -- python array (list) containing the dimensions 
                of each layer in our network

  Returns:
  parameters -- python dictionary containing your parameters
                {"W1", "b1", "W2", "b2", ... , "WL", "Wb"}:
                Wl -- weight matrix of shape 
                      (layers_dims[l], layers_dims[l-1])
                bl -- bias vector of shape
                      (layer_dims[l], 1)
  """
  parameters = {}
  L = len(layer_dims)
  for l in range(1, L):
    parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
    parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
    assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
  
  return parameters

def he_initialize_parameters_deep(layer_dims):
  """
  Arguments:
  layer_dims -- python array (list) containing the dimensions 
                of each layer in our network

  Returns:
  parameters -- python dictionary containing your parameters
                {"W1", "b1", "W2", "b2", ... , "WL", "Wb"}:
                Wl -- weight matrix of shape 
                      (layers_dims[l], layers_dims[l-1])
                bl -- bias vector of shape
                      (layer_dims[l], 1)
  """
  parameters = {}
  L = len(layer_dims)
  for l in range(1, L):
    parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
    parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
    assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
  
  return parameters

def linear_activation_forward(A_prev, W, b, activation):
  """
  Arguments:
  A_prev -- activations from previous layer (or input data): 
            (size of previous layer, number of examples)
  W --      weights matrix: numpy array of shape 
            (size of current layer, size of previous layer)
  b --      bias vector, numpy array of shape 
            (size of the current layer, 1)
  activation -- the activation to be used in this layer, 
                stored as a text string: "sigmoid" or "relu"

  Returns:
  A --      the output of the activation function, 
            also called the post-activation value 
  cache --  a python tuple containing 
            "linear_cache" and "activation_cache"
  """
  if activation == "sigmoid":
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = sigmoid(Z)
  elif activation == "relu":
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = relu(Z)

  cache = (linear_cache, activation_cache)

  return A, cache

parameters = initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

def L_model_forward(X, parameters):
  """
  Arguments:
  X --      data, numpy array of shape (input size, number of examples)
  parameters -- output of initialize_parameters_deep()
  
  Returns:
  AL --     last post-activation value
  caches -- list of caches containing:
            every cache of linear_activation_forward() 
            (there are L-1 of them, indexed from 0 to L-1)
  """
  caches = []
  A = X
  L = len(parameters) // 2

  for l in range(1, L):
    A_prev = A
    A, cache = linear_activation_forward(
      A_prev, 
      parameters['W' + str(l)], 
      parameters['b' + str(l)],
      'relu'
    )
    caches.append(cache)
  AL, cache = linear_activation_forward(
    A,
    parameters['W' + str(L)],
    parameters['b' + str(L)],
    'sigmoid'
  )
  caches.append(cache)

  return AL, caches


def L_model_forward_with_dropout(X, parameters, keep_prob = 0.5):
  """
  Arguments:
  X --      data, numpy array of shape (input size, number of examples)
  parameters -- output of initialize_parameters_deep()
  
  Returns:
  AL --     last post-activation value
  caches -- list of caches containing:
            every cache of linear_activation_forward() 
            (there are L-1 of them, indexed from 0 to L-1)
  """
  caches = []
  A = X
  L = len(parameters) // 2

  for l in range(1, L):
    A_prev = A
    A, cache = linear_activation_forward(
      A_prev, 
      parameters['W' + str(l)], 
      parameters['b' + str(l)],
      'relu'
    )
    D = np.random.rand(A.shape[0], A.shape[1])
    D = (D < keep_prob).astype(int)
    cache.append(D)
    A = np.multiply(A, D) / keep_prob

    caches.append(cache)
  AL, cache = linear_activation_forward(
    A,
    parameters['W' + str(L)],
    parameters['b' + str(L)],
    'sigmoid'
  )
  caches.append(cache)

  return AL, caches


def compute_cost(AL, Y):
  """
  Arguments:
  AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
  Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

  Returns:
  cost -- cross-entropy cost
  """
  m = Y.shape[1]
  cost = -(1/m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
  # To make sure your cost's shape is what we expect 
  # (e.g. this turns [[17]] into 17).
  cost = np.squeeze(cost)

  return cost

def linear_backward(dZ, cache):
  """
  Arguments:
  dZ --     Gradient of the cost with respect to the linear output 
            (of current layer l)
  cache --  tuple of values (A_prev, W, b) 
            coming from the forward propagation in the current layer

  Returns:
  dA_prev -- Gradient of the cost with respect to the activation 
            (of the previous layer l-1), same shape as A_prev
  dW --     Gradient of the cost with respect to W (current layer l), 
            same shape as W
  db --     Gradient of the cost with respect to b (current layer l), 
            same shape as b
  """
  A_prev, W, b = cache
  m = A_prev.shape[1]
  dW = (1/m) * np.dot(dZ, A_prev.T)
  db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
  dA_prev = np.dot(W.T, dZ)

  return dA_prev, dW, db

def relu_backward(dA, activation_cache):
  Z = activation_cache
  dZ = dA * ((Z > 0) * 1)
  return dZ

def sigmoid_backward(dA, activation_cache):
  Z = activation_cache
  dZ = dA * np.multiply(sigmoid(Z)[0],(1 - sigmoid(Z)[0]))
  return dZ

def linear_activation_backward(dA, cache, activation):
  """
  Arguments:
  dA --     post-activation gradient for current layer l 
  cache --  tuple of values (linear_cache, activation_cache) 
            we store for computing backward propagation efficiently
  activation -- the activation to be used in this layer, 
            stored as a text string: "sigmoid" or "relu"
  
  Returns:
  dA_prev -- Gradient of the cost with respect to the activation 
            (of the previous layer l-1), same shape as A_prev
  dW --     Gradient of the cost with respect to W (current layer l), 
            same shape as W
  db --     Gradient of the cost with respect to b (current layer l), 
            same shape as b
  """
  linear_cache, activation_cache = cache

  if activation == 'relu':
    dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
  elif activation == 'sigmoid':
    dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

  return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
  """
  Arguments:
  AL -- probability vector, output of the forward propagation (L_model_forward())
  Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
  caches -- list of caches containing:
              every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
              the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
  
  Returns:
  grads -- A dictionary with the gradients
            grads["dA" + str(l)] = ... 
            grads["dW" + str(l)] = ...
            grads["db" + str(l)] = ... 
  """
  grads = {}
  # number of layers:
  L = len(caches) 
  m = AL.shape[1]
  Y = Y.reshape(AL.shape)

  dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
  dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, caches[L - 1], 'sigmoid')
  grads['dA' + str(L - 1)] = dA_prev_temp
  grads['dW' + str(L)] = dW_temp
  grads['db' + str(L)] = db_temp

  for l in reversed(range(L - 1)):
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], caches[l], 'relu')
    grads['dA' + str(l)] = dA_prev_temp
    grads['dW' + str(l + 1)] = dW_temp
    grads['db' + str(l + 1)] = db_temp

  return grads

def update_parameters(params, grads, learning_rate):
  """
  Arguments:
  params -- python dictionary containing your parameters 
  grads -- python dictionary containing your gradients, output of L_model_backward
  
  Returns:
  parameters -- python dictionary containing your updated parameters 
                parameters["W" + str(l)] = ... 
                parameters["b" + str(l)] = ...
  """
  parameters = params.copy()
  L = len(parameters) // 2

  for l in range(L):
    parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - np.multiply(learning_rate, grads['dW' + str(l + 1)])
    parameters['W' + str(l + 1)] = parameters['b' + str(l + 1)] - np.multiply(learning_rate, grads['db' + str(l + 1)])

  return parameters

def compute_cost_with_regularization(AL, Y, parameters, lambd):

  L = len(parameters) // 2

  m = Y.shape[1]
  cross_entropy_cost = compute_cost(AL, Y)
  L2_regularization_cost = 0
  for l in range(L):
    L2_regularization_cost = L2_regularization_cost + np.sum(np.square(parameters['W' + str(l + 1)]))
  L2_regularization_cost =  L2_regularization_cost * lambd / (2 * m)
  cost = cross_entropy_cost + L2_regularization_cost

  return cost

def L_model_backward_with_regularization(AL, Y, caches, lambd):
  """
  Arguments:
  AL -- probability vector, output of the forward propagation (L_model_forward())
  Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
  caches -- list of caches containing:
              every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
              the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
  lambd -- regularization hyperparameter, scalar
  
  Returns:
  grads -- A dictionary with the gradients
            grads["dA" + str(l)] = ... 
            grads["dW" + str(l)] = ...
            grads["db" + str(l)] = ... 
  """
  grads = {}
  # number of layers:
  L = len(caches) 
  m = AL.shape[1]
  Y = Y.reshape(AL.shape)

  dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
  dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, caches[L - 1], 'sigmoid')
  grads['dA' + str(L - 1)] = dA_prev_temp
  grads['dW' + str(L)] = dW_temp
  grads['db' + str(L)] = db_temp

  for l in reversed(range(L - 1)):
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], caches[l], 'relu')
    grads['dA' + str(l)] = dA_prev_temp
    grads['dW' + str(l + 1)] = dW_temp
    grads['db' + str(l + 1)] = db_temp
  
  return grads

def L_model_backward_with_dropout(AL, Y, caches, keep_prob):
  """
  Arguments:
  AL -- probability vector, output of the forward propagation (L_model_forward())
  Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
  caches -- list of caches containing:
              every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
              the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
  keep_prob -- probability of keeping a neuron active during active drop-out, scalar
  
  Returns:
  grads -- A dictionary with the gradients
            grads["dA" + str(l)] = ... 
            grads["dW" + str(l)] = ...
            grads["db" + str(l)] = ... 
  """
  grads = {}
  # number of layers:
  L = len(caches) 
  m = AL.shape[1]
  Y = Y.reshape(AL.shape)

  dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
  dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, caches[L - 1], 'sigmoid')
  D = caches[L][4]
  grads['dA' + str(L - 1)] = (dA_prev_temp * D) / keep_prob
  grads['dW' + str(L)] = dW_temp
  grads['db' + str(L)] = db_temp

  for l in reversed(range(L - 1)):
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], caches[l], 'relu')
    D = caches[l][4]
    grads['dA' + str(l)] = (dA_prev_temp * D) / keep_prob
    grads['dW' + str(l + 1)] = dW_temp
    grads['db' + str(l + 1)] = db_temp
  
  return grads

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False, lambd = 0, keep_prob = 1):
  """
  Arguments:
  X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
  Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
  layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
  learning_rate -- learning rate of the gradient descent update rule
  num_iterations -- number of iterations of the optimization loop
  print_cost -- if True, it prints the cost every 100 steps
  
  Returns:
  parameters -- parameters learnt by the model. They can then be used to predict.
  """
  costs = []

  parameters = initialize_parameters_deep(layers_dims)
  for i in range(0, num_iterations):

    if keep_prob == 1:
      AL, caches = L_model_forward(X, parameters)
    elif keep_prob < 1:
      AL, caches = L_model_forward_with_dropout(X, parameters)
     
    if lambd == 0:
      cost = compute_cost(AL, Y)
    else:
      cost = compute_cost_with_regularization(AL, Y, parameters, lambd)
    
    assert(lambd == 0 or keep_prob == 1)

    if lambd == 0 and keep_prob == 1:
      grads = L_model_backward(AL, Y, caches)
    elif lambd != 0:
      grads = L_model_backward_with_regularization(AL, Y, caches)
    elif keep_prob < 1:
      grads = L_model_backward_with_dropout(AL, Y, caches, keep_prob)

    parameters = update_parameters(parameters, grads, learning_rate)

    if print_cost and i % 100 == 0 or i == num_iterations - 1:
      print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
    if i % 100 == 0 or i == num_iterations:
      cost.append(cost)
  
  return parameters, costs


