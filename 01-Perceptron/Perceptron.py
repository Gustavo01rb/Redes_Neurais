import sys
import numpy as np
from tabulate import tabulate
import pandas as pd
sys.path.append('..')
from typing import Callable
from utils.Activation_Function import ActivationFunction as AF

class Perceptron:
    def __init__(
        self,
        epoch: int = 50,
        learning_rate: float = 0.2,
        actv_function: Callable = AF.step_function_1,
        bias: int = -1,
        tol: float = 0.02
    ) -> None:
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.actv_function = actv_function
        self.weights = None
        self.bias = bias
        self.tol = tol
        self.history = {
            "errors": [],
            "weights": []
        }
    """
        Inicializa um objeto Perceptron.

        Parâmetros:
        - epoch (int): Número máximo de épocas para treinamento (padrão: 50).
        - learning_rate (float): Taxa de aprendizado (padrão: 0.2).
        - actv_function (Callable): Função de ativação (padrão: AF.step_function_1).
        - bias (int): Valor do bias/viés (padrão: -1).
        - tol (float): Tolerância para considerar a convergência (padrão: 0.02).
    """

    def __include_bias(self, x):
        bias = np.repeat(self.bias, x.shape[0])
        x = np.insert(x, 0, bias, axis=1)
        return x
    """
        Inclui o bias nos dados de entrada.

        Parâmetros:
        - x: Dados de entrada.

        Retorna:
        - x com o bias incluído.
    """

    def __init_weights(self, number):
        self.weights = np.random.uniform(-0.5, 0.5, number)
    """
        Inicializa os pesos.

        Parâmetros:
        - number: Número de pesos a serem criados.
    """

    def fit(self, X, Y):
        X = self.__include_bias(X)
        self.__init_weights(X.shape[1])
        for current_epoch in range(self.epoch):
            erro = False
            current_erro = 0
            for index_current_sample, current_sample in enumerate(X):
                u = np.dot(self.weights, current_sample)
                y = self.actv_function(u)
                
                if y == Y[index_current_sample]:
                    continue

                erro = True
                current_erro += abs(Y[index_current_sample] - y)
                self.weights += self.learning_rate * (Y[index_current_sample] - y) * current_sample
            current_erro /= X.shape[0]
            self.history["errors"].append(current_erro)
            self.history["weights"].append(self.weights.copy())
            if not erro or current_erro <= self.tol:
                self.epoch = current_epoch
                return
    """
        Realiza o treinamento do Perceptron.
        Parâmetros:
        - X: Dados de entrada para treinamento.
        - Y: Saídas esperadas para os dados de entrada.

        Retorna:
        - None
    """

    def predict(self, sample):
        sample = np.insert(sample, 0, self.bias)
        return self.actv_function(np.dot(self.weights, sample))
    """
        Realiza a predição para uma amostra.

        Parâmetros:
        - sample: Amostra a ser classificada.

        Retorna:
        - Saída do Perceptron para a amostra.
    """

    def show_info(self, title):
        print("\n\n", title)
        print("\t Valor do bias (viés): ", self.bias)
        print("\t Número de épocas até convergir: ", self.epoch)
        print("\t Valor da taxa de aprendizado: ", self.learning_rate)
        print("\t Histórico de erros: ")
        for index, error in enumerate(self.history['errors']):
            print("\t\t Época", index, ":", error)
        print("\t Histórico de pesos:")
        self.history['weights'] = pd.DataFrame(self.history['weights'], columns=['W' + str(i) for i in range(self.weights.shape[0])])
        print(tabulate(self.history['weights'], headers='keys', tablefmt='psql'))
    """
        Mostra informações sobre o Perceptron.

        Parâmetros:
        - title: Título a ser exibido.

        Retorna:
        - None
    """