import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from utils import ActivationFunction as AF

"""
    Implementação de um Perceptron Multicamadas (MLP) para classificação.

    Parâmetros:
    - dims (list): Lista de inteiros representando o número de nós em cada camada, incluindo as camadas de entrada e saída.
    - eta (float): Taxa de aprendizado para atualização dos pesos durante o treinamento.
    - activation (str): Função de ativação a ser utilizada nas camadas ocultas ('sigmoid', 'linear', 'relu', 'tanh').
    - stochastic (float): Parâmetro de estocasticidade para atualização dos pesos. O valor padrão é 0.0 (sem estocasticidade).
    - max_epochs (int): Número máximo de épocas para o treinamento.
    - deltaE (float): Limiar de convergência para interrupção antecipada baseada na mudança do erro.
    - alpha (float): Parâmetro de momento para a atualização dos pesos. O valor padrão é 0.8.

    Métodos:
    - add_ones_column(X): Adiciona uma coluna de 1s ao início da matriz X.
    - fit(X, y, Xtest=None, ytest=None, weights=None): Realiza o treinamento da MLP com os dados de entrada e saída fornecidos.
    - predict(X): Realiza a predição com base nos dados de entrada fornecidos.
    - score(X, y): Calcula a pontuação da MLP com base nos dados de entrada e saída fornecidos.
    - _RMSE(y, yp): Calcula o erro quadrático médio entre o valor real y e o valor predito yp.
    - _forwardpass(X, weights): Realiza a passagem direta (forward pass) dos dados pela MLP.
    - _backwardpass(y, Y, x, u, weights): Realiza a passagem reversa (backward pass) para atualização dos pesos.
"""

class MLP:
    def __init__(self, dims=[1, 2, 4, 1], eta=0.001, activation='sigmoid', stochastic=0.0,
                 max_epochs=10000, deltaE=-np.inf, alpha=0.8):
        self.dims = dims
        self.eta = eta
        self.activation = activation
        self.stochastic = stochastic
        self.max_epochs = max_epochs
        self.deltaE = deltaE
        self.alpha = alpha
        self.dW = []
        if activation == 'sigmoid':
            self.f = AF.sigmoidal_function
            self.df = AF.dsigmoidal_function
        elif activation == 'linear':
            self.f = lambda x: x
            self.df = lambda x: 1
        elif activation == 'relu':
            self.f = AF.relu
            self.df = AF.drelu
        elif activation == 'tanh':
            self.f = AF.tanh
            self.df = AF.dtanh
        else:
            raise ValueError("Invalid activation function: %r" % activation)
    """
        Inicializa a MLP com os parâmetros fornecidos.

        Args:
        - dims (list): Lista de inteiros representando o número de nós em cada camada, incluindo as camadas de entrada e saída.
        - eta (float): Taxa de aprendizado para atualização dos pesos durante o treinamento.
        - activation (str): Função de ativação a ser utilizada nas camadas ocultas ('sigmoid', 'linear', 'relu', 'tanh').
        - stochastic (float): Parâmetro de estocasticidade para atualização dos pesos. O valor padrão é 0.0 (sem estocasticidade).
        - max_epochs (int): Número máximo de épocas para o treinamento.
        - deltaE (float): Limiar de convergência para interrupção antecipada baseada na mudança do erro.
        - alpha (float): Parâmetro de momento para a atualização dos pesos. O valor padrão é 0.8.
    """

    def add_ones_column(self, X):
        a, b = X.shape
        Xt = np.zeros([a, b + 1])
        Xt[:, 1:] = X
        Xt[:, 0] = np.ones(a)
        return Xt
    """
        Adiciona uma coluna de 1s ao início da matriz X.

        Args:
        - X (numpy.ndarray): Matriz de entrada.

        Returns:
        - numpy.ndarray: Matriz X com uma coluna de 1s adicionada ao início.
    """

    def fit(self, X, y, Xtest=None, ytest=None, weights=None):
        if X.shape[0] != y.shape[0]:
            raise ValueError("Training and target shapes don't match")
        self.weights = weights
        if self.weights is None:
            self.weights = []
            for i in range(len(self.dims) - 1):
                W = np.random.rand(self.dims[i + 1], self.dims[i] + 1) - 0.5
                self.weights.append(W)
        self.dW = [np.zeros(W.shape) for W in self.weights]
        self.train_error = np.zeros(self.max_epochs + 1)
        self.test_error = np.zeros(self.max_epochs + 1)
        self.train_error[-1] = np.infty
        self.test_error[-1] = np.infty
        t = 0
        while t < self.max_epochs:
            if self.stochastic:
                idx = np.random.permutation(X.shape[0])
                Xs, ys = X[idx], y[idx]
            else:
                Xs, ys = X, y
            Y, x, u = self._forward_pass(Xs, self.weights)
            rmse = self._RMSE(Y, ys)
            self.train_error[t] = rmse
            if Xtest is not None:
                rmse = self.score(Xtest, ytest)
                self.test_error[t] = rmse
                delta = self.test_error[t] - self.test_error[t - 1]
            else:
                delta = self.train_error[t] - self.train_error[t - 1]
            if abs(delta) < self.deltaE:
                break
            # Backward pass
            self.weights = self._backward_pass(ys, Y, x, u, self.weights)
            t += 1
        self.train_error = self.train_error[:t]
        self.test_error = self.test_error[:t]
        return self.weights
    """
        Realiza o treinamento da MLP com os dados de entrada e saída fornecidos.

        Args:
        - X (numpy.ndarray): Matriz de entrada.
        - y (numpy.ndarray): Vetor de saída.
        - Xtest (numpy.ndarray): Matriz de entrada para teste. Opcional.
        - ytest (numpy.ndarray): Vetor de saída para teste. Opcional.
        - weights (numpy.ndarray): Vetor de pesos iniciais. Opcional.

        Returns:
        - MLP: Próprio objeto MLP.
    """

    def predict(self, X):
        Y, _, _ = self._forward_pass(X, self.weights)
        return Y
    """
        Realiza a predição com base nos dados de entrada fornecidos.

        Args:
        - X (numpy.ndarray): Matriz de entrada.

        Returns:
        - numpy.ndarray: Vetor de saída predito.
    """

    def score(self, X, y):
        yp = self.predict(X)
        return self._RMSE(y, yp)
    """
        Calcula a pontuação da MLP com base nos dados de entrada e saída fornecidos.

        Args:
        - X (numpy.ndarray): Matriz de entrada.
        - y (numpy.ndarray): Vetor de saída.

        Returns:
        - float: Pontuação da MLP.
    """

    def _RMSE(self, y, yp):
        return np.sqrt(np.sum((yp - y) ** 2) / y.shape[0])
    """
        Calcula o erro quadrático médio entre o valor real y e o valor predito yp.

        Args:
        - y (numpy.ndarray): Vetor de valores reais.
        - yp (numpy.ndarray): Vetor de valores preditos.

        Returns:
        - float: Erro quadrático médio.
    """

    def _forward_pass(self, X, weights):
        Y = X
        x = []  
        u = []  
        for i in range(len(weights)):
            X = self.add_ones_column(Y)
            x.append(X)  
            U = np.dot(X, weights[i].T) 
            u.append(U)  
            Y = self.f(U)  
        return Y, x, u
    """
        Realiza a passagem direta (forward pass) dos dados pela MLP.

        Args:
        - X (numpy.ndarray): Matriz de entrada.
        - weights (list): Lista de matrizes de pesos.

        Returns:
        - numpy.ndarray: Vetor de saída da MLP.
    """

    def _backward_pass(self, y, Y, x, u, weights):
        D = -self.df(u[-1]) * (y - Y)  
        delta = [D]
        for i in range(len(weights) - 1):
            W = weights[::-1][i]  
            U = u[::-1][i + 1]  
            d = self.df(U) * (delta[i] @ W)[:, 1:]
            delta.append(d)
        delta.reverse()  
        weights_new = []
        for i in range(len(weights)):
            W = weights[i]
            momentum = self.alpha * self.dW[i]
            learning_term = self.eta * (delta[i].T @ x[i])
            Wnew = W - learning_term + momentum
            self.dW[i] = Wnew - W
            weights_new.append(Wnew)
        return weights_new

    def __repr__(self):
        return "in:%i, hidden:%i out:%i " % tuple(self.dims)
    """
        Realiza a passagem reversa (backward pass) para atualização dos pesos.

        Args:
        - y (float): Saída da MLP.
        - Y (float): Valor esperado de saída.
        - x (numpy.ndarray): Vetor de entrada.
        - u (list): Lista de matrizes de ativação.
        - weights (list): Lista de matrizes de pesos.

        Returns:
        - numpy.ndarray: Vetor de atualização de pesos.
    """