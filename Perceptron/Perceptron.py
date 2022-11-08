import sys
import numpy as np
from tabulate import tabulate
import pandas as pd
sys.path.append('..')
from typing import Callable
import matplotlib.pyplot as plt
from utils.Activation_Function import Activation_function as AF

class Perceptron:
    def __init__(
        self,
        epoch         : int      = 50,
        learning_rate : float    = 0.2,
        actv_function : Callable = AF.step_function_1,
        bias          : int      = -1,
        tol           : float    = 0.02
    ) -> None:
        self.epoch         = epoch
        self.learning_rate = learning_rate
        self.actv_function = actv_function
        self.weights       = list()
        self.bias          = bias
        self.tol           = tol
        self.history       = {
            "errors"  : list(),
            "weights" : list()
        }
    

    def __include_bias(self,x):
        bias = np.repeat(self.bias, x.shape[0])
        x = np.insert(x, 0, bias, axis=1)
        return x 

    def __init_weights(self, number):
        '''
            Função para iniciar os valore dos pesos.
            O parâmetro number indica a quantidade de pesos a ser criada.
            Os pesos serão gerados no intervalo: [-0,5;0,5]
        '''
        self.weights = np.random.uniform(-0.5,0.5,number)        
        self.weights = np.array(self.weights)

    def fit(self,X,Y):
        X = self.__include_bias(X)                  # Determinar todos os X de K
        self.__init_weights(X.shape[1])             # Iniciar os pesos com valores aleatórios
        for current_epoch in range(self.epoch):
            erro = False                            #Erro inexistente
            current_erro = 0
            for index_current_sample, current_sample in enumerate(X[:]):
                u = np.dot(self.weights, current_sample)
                y = self.actv_function(u)
                
                if y == Y[index_current_sample]: continue

                erro = True
                current_erro += abs(Y[index_current_sample] - y)
                for index_weight in range(self.weights.shape[0]):
                    self.weights[index_weight] += self.learning_rate * (Y[index_current_sample] - y) * current_sample[index_weight]
            current_erro = current_erro/X.shape[0]
            self.history["errors"].append(current_erro)
            self.history["weights"].append(self.weights.copy())
            if not erro or current_erro <= self.tol:
                self.epoch = current_epoch
                return

    def predict(self, sample):
        sample = np.insert(sample, 0, self.bias)
        return self.actv_function(np.dot(self.weights, sample))
    
    def show_info(self, title):
        print("\n\n",title)
        print("\t Valor do bias(viés): ", self.bias)
        print("\t Número de épocas até convergir: ", self.epoch)
        print("\t Valor da taxa de aprendizado: ", self.learning_rate)
        print("\t Histórico de erros: \n\t\t", self.history['errors'])
        print("\t Histórico de pesos:")
        self.history['weights'] = pd.DataFrame(self.history['weights'], columns=['W' + str(i) for i in range(self.weights.shape[0])])
        print(tabulate(self.history['weights'], headers = 'keys', tablefmt = 'psql'))