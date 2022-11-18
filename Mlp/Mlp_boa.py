import sys
import numpy as np
import pandas as pd
sys.path.append('..')
from typing import Callable
import matplotlib.pyplot as plt
from utils.Activation_Function import Activation_function as AF

class MLP:
    def __init__(
            self,
            tol             : float    = 0.000000001,   # Tolerância
            bias            : int      = -1,            # Valor do Bias
            epoch           : int      = 10500,         # Número máximo de épocas
            layers          : int      = [3,1],         # Quantidade de neurônios em cada camada. Não incluso camada de entrada
            learning_rate   : float    = 0.2,           # Taxa de aprendizado
            actv_function   : Callable = AF.sigmoidal_function, # Função de Ativação
            d_actv_function : Callable = AF.dsigmoidal_function, # Derivada da função de ativação
        ) -> None:
        self.tol             = tol
        self.bias            = bias
        self.epoch           = epoch
        self.layers          = list()
        self.layers_info     = layers
        self.learning_rate   = learning_rate
        self.actv_function   = actv_function
        self.d_actv_function = d_actv_function 
        self.erro = [1+tol]

    def __init_weights(self, X):
        '''
            Essa função tem como objetivo gerar a matriz de pesos de uma MLP. O pesos serão iniciados com valores aleatórios
            no intervalo definido: [-0.5;0.5]. Podemos considerar os pesos em uma estrura de 3 dimensões:
                1 dimesão       -> Camada pertecente do peso
                2 e 3 dimansões -> Matriz de pesos da camada
            Os pesos W(i,j) são definidos da seguinte maneira
                i -> neurônio de destino
                j -> neurônio de origem 
        '''
        input_layer_size = X.shape[1]
        self.layers_info.insert(0,input_layer_size)
        for index, n_neurons in enumerate(self.layers_info):
            self.layers.append(np.array([np.random.uniform(-0.5,0.5,n_neurons+1) for _ in range(self.layers_info[index+1])]))
            if index == len(self.layers_info)-2:
                break

    def __include_bias(self, sample):
        sample = np.array(sample)
        return np.insert(sample, 0, self.bias)
    
    def forwardpass(self, sample, weights):
        '''
            Essa função recebe como parâmetro uma amostra (valores da camada de entrada) -> sample
            Recebe um tensor de pesos que nesse caso possui 3 dimensões
            u = true output neuron
            y = f(output)
        '''
        
        input_layer  = [sample]
        activation   = list()
        for weight_layer in weights:         
            u  = [np.dot(weight_neuron, input_layer[-1]) for weight_neuron in weight_layer] 
            activation.append(np.array(u))

            y  = np.array([self.actv_function(out) for out in u])
            input_layer.append(self.__include_bias(y))

            Y = self.actv_function(np.array(u))
        return Y, activation, input_layer


    def backwardpass(self, expected_out, true_out, input_values, active_values, weights):
        D = -self.d_actv_function(active_values[-1])*(expected_out - true_out)  
        delta = [D]
        for i in range(len(weights) -1):
            W = weights[::-1][i]         # Inverter a matriz de pesos
            U = active_values[::-1][i+1] # Inverter a matriz dos valore de ativação
            d = -self.d_actv_function(U) * (delta[i]@W).sum()
            delta.append(d)
        delta.reverse() 

        for i, weight_layer in enumerate(weights):
            for j, line in enumerate(weight_layer):
                for c in range(len(line)):
                    weights[i][j][c] += self.learning_rate * delta[i][j] * input_values[+1][i]                
        return weights

    def fit(self, X, Y):
        '''
            -> Para os parâmetros X e Y espera-se uma matriz ou um vetor coluna:
                x = |A|X1|X2|  Y = |A|Y1|   "Valores ilustrativos"
                    |0|00|00|      |0|00|
                    |1|00|01|      |1|01|
                    |2|01|00|      |2|01|
                    |3|01|01|      |3|00|
            -> Importante: Para o melhor funcionamento do método espera-se que as listas sejam np.array's
        '''
        self.__init_weights(X)

        for current_epoch in range(self.epoch):
            self.erro.append(self.erro[-1])
            for index_current_sample, current_sample in enumerate(X[:]):
                current_sample  = self.__include_bias(current_sample)
                expected_out    = Y[index_current_sample]
                
                true_out, active_neuron, input_values = self.forwardpass(current_sample, self.layers)
                self.layers = self.backwardpass(expected_out,true_out, input_values, active_neuron, self.layers)
                
    
    def show_weights(self):
        print("\nPesos")
        for layer, l in enumerate(self.layers):
            print("\nCamada: ", layer)
            for line in l:
                print("\t",line)
    def predict(self, x):
        x = self.__include_bias(x)
        Y, _, __ = self.forwardpass(x, self.layers)
        return Y 

xor_truth_table = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]   
xor_truth_table = pd.DataFrame(xor_truth_table, columns=['X1','X2','Y'])
x = np.array([xor_truth_table['X1'], xor_truth_table['X2']]).T
y = np.array([xor_truth_table['Y']]).T 
model =  MLP( layers=[4,2,1], epoch=500, learning_rate=0.2)
model.fit(x,y)

X = np.linspace(-0.5, 1.5, 100)
Y = np.linspace(-0.5, 1.5, 100)
X, Y = np.meshgrid(X, Y)
def F(x,y):
    return model.predict(np.array([x,y]))
Z = np.vectorize(F)(X,Y)
plt.pcolor(X,Y,Z, cmap='RdBu')
plt.colorbar()
cntr = plt.contour(X,Y,Z, levels = [0.5])
plt.clabel(cntr, inline=1, fontsize=10)
plt.scatter([0,1], [0,1], s = 500, c = 'r')
plt.scatter([1,0], [0,1], s = 500, marker = 'v')
plt.grid()
plt.show()