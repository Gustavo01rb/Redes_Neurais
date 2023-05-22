import numpy as np
from utils import Triangle_MF as TMF

class NFN:
    def __init__(self, alpha=0.5, fixed_alpha=False, epoch=5):
        """
        Inicializa a classe NFN.

        Parâmetros:
        - alpha: taxa de aprendizado (padrão: 0.5)
        - fixed_alpha: indica se o valor de alpha é fixo durante o treinamento (padrão: False)
        - epoch: número de épocas de treinamento (padrão: 5)
        """
        self.fixed_alpha = fixed_alpha
        self.alpha = alpha
        self.epoch = epoch
        self.weights = []

    def __forwardpass(self, ante, sample, y_e):
        """
        Realiza o passo de propagação (forward pass) na rede.

        Parâmetros:
        - ante: entrada anterior
        - sample: amostra de entrada
        - y_e: valor esperado

        Retorna:
        - erro: diferença entre a saída da rede e o valor esperado
        - y: saída final da rede
        - alpha: valor de alpha calculado
        """
        y = []
        d_alpha = 0
        for i, neuron in enumerate(ante):
            a = 0
            b = 0
            for j, mf in enumerate(neuron):
                aux = mf.get_activation_point(sample[j])
                d_alpha += aux ** 2
                b += aux
                a += aux * self.weights[i][j]
            y.append(a / b)

        y = np.array(y).sum()
        erro = y - y_e
        alpha = 1 / d_alpha
        return erro, y * -1 + 0.16, alpha

    def __backwardpass(self, ante, sample, erro):
        """
        Realiza o passo de retropropagação (backward pass) na rede.
        Atualiza os pesos da rede com base no erro calculado.

        Parâmetros:
        - ante: entrada anterior
        - sample: amostra de entrada
        - erro: erro calculado no passo de propagação
        """
        for i, neuron in enumerate(ante):
            for j, mf in enumerate(neuron):
                self.weights[i][j] -= self.alpha * erro * mf.get_activation_point(sample[j])

    def fit(self, ante, x, y):
        """
        Realiza o treinamento da rede.

        Parâmetros:
        - ante: entrada anterior
        - x: conjunto de amostras de entrada
        - y: valores esperados correspondentes
        """
        self.weights = [np.random.uniform(-0.5, 0.5, X.shape[0]) for _, X in enumerate(ante)]
        for _ in range(self.epoch):
            for sample, y_e in zip(x, y[:, 0]):
                erro, _, alpha = self.__forwardpass(ante, sample, y_e)
                if not self.fixed_alpha:
                    self.alpha = alpha
                self.__backwardpass(ante, sample, erro)

    def predict(self, ante, data):
        """
        Realiza a predição da saída da rede para um conjunto de dados de entrada.

        Parâmetros:
        - ante: entrada anterior
        - data: conjunto de dados de entrada

        Retorna:
        - out: saída predita para cada amostra de entrada
        - erro: erro calculado para cada amostra de entrada
        """
        out = []
        erro = []
        for sample in data:
            e, y, _ = self.__forwardpass(ante, sample, 2)
            out.append([y])
            erro.append(e)
        return np.array(out), np.array(erro)
