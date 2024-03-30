import numpy as np
class utilactivation:
    def relu(x):
        return np.maximum(0,x)
    def tanh(x):
        return np.tanh(x)
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def lrelu(x):
        return np.maximum(0.01 * x, x)
