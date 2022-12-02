import numpy as np
from helpers import triangle_mf as TMF

class NFN:
    def __init__(self, alpha : float = 0.5, fixed_alpha : bool = False, epoch : int = 5) -> None:
        self.fixed_alpha = fixed_alpha
        self.alpha  = alpha
        self.epoch  = epoch
        self.weights = list()

    def __forwardpass(self, ante : np.ndarray, sample : np.ndarray, y_e : float):
        y = list()
        d_alpha = 0
        for i, neuron in enumerate(ante):
            a = 0
            b = 0
            for j, mf in enumerate(neuron):
                aux = sample[j] * mf.get_activation_point(sample[j])
                #d_alpha += mf.get_activation_point(sample[j])/ sample[j]**2  
                d_alpha += mf.get_activation_point(sample[j])**2 
                b += aux
                a += aux * self.weights[i][j]
            y.append(a/b)

        y = np.array(y).sum()
        erro =  y_e - y
        alpha = 1 / d_alpha
        return erro, y, alpha
        

    def __backwardpass(self, ante : np.ndarray, sample : np.ndarray, erro : float) -> None:
        for i, neuron in enumerate(ante):
            for j, mf in enumerate(neuron):
                #print(f"Peso: {self.weights[i][j]}, Alpha: {self.alpha}, erro: {erro}, ac: {mf.get_activation_point(sample[j])}, novo valor: {self.alpha * erro * mf.get_activation_point(sample[j])}")
                self.weights[i][j] += self.alpha * erro * mf.get_activation_point(sample[j])

        
    
    
    
    def fit(self, ante : np.ndarray, x : np.ndarray, y : np.array) -> None:
        self.weights = np.array([np.random.uniform(-0.5,0.5, X.shape[0]) for _ , X in enumerate(ante)])
        for current_epoch in range(self.epoch):

            for i, sample in enumerate(x):
                y_e = y[i][0]
                erro, _ , alpha = self.__forwardpass(ante, sample, y_e)
                if not self.fixed_alpha: self.alpha = alpha
                self.__backwardpass(ante, sample, erro)

    def predict(self, ante, data):
        out = list()
        erro = list()
        for sample in data:
            e,y,_ = self.__forwardpass(ante,sample,2)
            out.append([y])
            erro.append(e)
        return np.array(out), np.array(erro)