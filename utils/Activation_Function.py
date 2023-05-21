import numpy as np

class ActivationFunction:
    @staticmethod
    def step_function_1(x):
        return np.where(x >= 0, 1, 0)

    @staticmethod
    def step_function_2(x):
        return np.where(x >= 0, 1, -1)

    @staticmethod
    def step_function_3(x):
        return np.where(x > 0, 1, np.where(x == 0, 0, -1))

    @staticmethod
    def linear_function(x):
        return np.where(x > 1, 1, np.where(x < 0, 0, x))

    @staticmethod
    def linear_function_no_saturation(x, a):
        return a * x

    @staticmethod
    def sigmoidal_function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoidal_function(x):
        sigmoid = ActivationFunction.sigmoidal_function(x)
        return sigmoid * (1 - sigmoid)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def dtanh(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def drelu(x):
        return np.where(x > 0, 1, 0)
