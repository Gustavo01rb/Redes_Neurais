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
            bias            : int      = 1,      # Valor do Bias
            epoch           : int      = 10500,     # Número máximo de épocas
            layers          : int      = [3,1],   # Quantidade de neurônios em cada camada. Não incluso camada de entrada
            learning_rate   : float    = 0.2,     # Taxa de aprendizado
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

    
    def __include_bias(self,x):
        '''
            Essa função tem como objetivo adicionar uma entrada, um viés ou bias (X0) para o conjunto de entrada
            de forma que:
                  Antes              Depois  
            x = |A|X1|X2|     x = |A|X0|X1|X2| 
                |0|00|00|         |0|-1|00|00| 
                |1|00|01|         |1|-1|00|01| 
                |2|01|00|         |2|-1|01|00| 
                |3|01|01|         |3|-1|01|01| 
        '''
        bias = np.repeat(self.bias, x.shape[0])
        x    = np.insert(x, 0, bias, axis=1)
        return x 

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
            self.layers.append(np.array([np.random.uniform(-0.5,0.5,n_neurons) for _ in range(self.layers_info[index+1])]))
            if index == len(self.layers_info)-2:
                break
    
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
        X = self.__include_bias(X)
        self.__init_weights(X)

        for current_epoch in range(self.epoch):
            self.erro.append(self.erro[-1])
            current_errors  = list()
            output = 0
            for index_current_sample, current_sample in enumerate(X[:]):
                output_layer    = list()                    # Variável que armazena os resultados obtidos de cada neurônio (resultado, actv(resultado))
                expected_out    = Y[index_current_sample]   # Saída desejada
                input_layer     = [current_sample]          # Definindo a camada de entrada de acordo com os valores de X

                for weight_layer in self.layers:         # Caminhando nos pesos de cada camada
                    out_neuron  = [np.dot(weight_neuron, input_layer[-1]) for weight_neuron in weight_layer] # Definindo a saída dos neurônios da camada
                    input_layer.append(np.array([self.actv_function(out) for out in out_neuron])) 
                    output_layer.append(np.array(out_neuron))
                current_errors.append(((np.array([expected_out]) - np.array(output_layer[-1]))**2).sum() / 2)
                

                # Back-Propagation
                input_layer  = list(reversed(input_layer))
                output_layer = list(reversed(output_layer))
                self.layers = list(reversed(self.layers))
                for index_back, weight_layer in enumerate(self.layers):
                    
                    if index_back == len(self.layers) -1 : break
                    if index_back == 0:
                        delta = [(expected_out - input_layer[0]) * self.d_actv_function(output_layer[0] )] 
                    else:
                        delta.append(((delta[-1]*weight_layer).sum() * self.d_actv_function(output_layer[index_back]) * -1 ))
                    for j, line in enumerate(weight_layer):
                        for i in range(len(line)):
                            self.layers[index_back][j][i] += self.learning_rate * delta[index_back][j] * input_layer[index_back+1][i]
                self.layers = list(reversed(self.layers))
            
                output = [current_sample] 
                for weight_layer in self.layers:         # Caminhando nos pesos de cada camada
                    out_neuron  = [np.dot(weight_neuron, output[-1]) for weight_neuron in weight_layer] # Definindo a saída dos neurônios da camada
                    output.append(np.array([self.actv_function(out) for out in out_neuron])) 
                output = output[-1]
            self.erro.append(np.array(current_errors).sum() / X.shape[0])
            if abs(self.erro[-1] - self.erro[-2])  <= self.tol :
                self.epoch = current_epoch
                print("Convergiu: ", self.epoch)
                return
    
    def show_weights(self):
        print("\nPesos")
        for layer, l in enumerate(self.layers):
            print("\nCamada: ", layer)
            for line in l:
                print("\t",line)
    def predict(self, x):
        input_list = [x]
        for weight_layer in self.layers:         # Caminhando nos pesos de cada camada
            out_neuron  = [np.dot(weight_neuron, input_list[-1]) for weight_neuron in weight_layer] # Definindo a saída dos neurônios da camada
            input_list.append(np.array([self.actv_function(out) for out in out_neuron])) 
        return np.array(input_list[-1])







xor_truth_table = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]   
xor_truth_table = pd.DataFrame(xor_truth_table, columns=['X1','X2','Y'])
x = np.array([xor_truth_table['X1'], xor_truth_table['X2']]).T
y = np.array([xor_truth_table['Y']]).T 
model =  MLP(layers=[5,2,1], epoch=5000, learning_rate=0.55)
model.fit(x,y)


X = np.linspace(-0.5, 1.5, 100)
Y = np.linspace(-0.5, 1.5, 100)
X, Y = np.meshgrid(X, Y)
def F(x,y):
    return model.predict(np.array([-1,x,y]))
Z = np.vectorize(F)(X,Y)
plt.pcolor(X,Y,Z, cmap='RdBu')
plt.colorbar()
cntr = plt.contour(X,Y,Z, levels = [0.5])
plt.clabel(cntr, inline=1, fontsize=10)
plt.scatter([0,1], [0,1], s = 500, c = 'r')
plt.scatter([1,0], [0,1], s = 500, marker = 'v')
plt.grid()
plt.show()
