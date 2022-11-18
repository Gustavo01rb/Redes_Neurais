import numpy as np

class Membership_Function:
    @staticmethod
    def triangle(x,a,m,b):
        y = np.zeros(x.shape[0])                      # Definindo um array de saída do tamnho da entrada.
        first_half = np.logical_and(a < x, x<=m)      # Definindo o intervalo de 'subida' da função.
        y[first_half] = (x[first_half]-a) / (m-a)     # Definindo os valores da saída para o intervalo de 'subida'.
        second_half = np.logical_and(m <= x, x < b)   # Definindo o intervalo de 'descida' da função.
        y[second_half] = (b - x[second_half]) / (b-m) # Definindo os valores da saída para o intervalo de 'descida'.
        return y
    
    @staticmethod
    def trapezoidal(x,a,m,n,b):
        y = np.zeros(x.shape[0])                      # Definindo saída do tamaho da entrada.
        first_part = np.logical_and( a < x, x <= m )  # Definindo o intervalo de subida.
        y[first_part] = (x[first_part] - a) / (m-a)   # Definindo os valores da saída no intervalo de subida.
        second_part = np.logical_and(m < x, x < n)    # Definindo o intervalo entre subida e decida.
        y[second_part] = 1                            # Definindo o valor 1 para todo o intervalo entre subida e decida.
        third_part = np.logical_and(n <= x, x < b)    # Definindo o intervalo de decida.  
        y[third_part] = (b - x[third_part]) / (b-n)   # Defininido os valores de saída para o intervalo de saída.
        return y
    
    @staticmethod
    def gaussian(x,k,m):
        k = k/2
        expoent = (-1)*((x-m)**2)/(k**2)
        return np.exp( expoent )