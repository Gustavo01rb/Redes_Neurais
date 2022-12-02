import numpy as np

class Triangle_MF:
    def __init__(self, range: np.linspace, a : float, m : float, b : float) -> None:
        self.range = range
        self.a = a
        self.m = m
        self.b = b
        y = np.zeros(range.shape[0])                      
        first_half = np.logical_and(a < range, range<=m)  
        y[first_half] = (range[first_half]-a) / (m-a)     
        second_half = np.logical_and(m <= range, range < b)   
        y[second_half] = (b - range[second_half]) / (b-m) 
        self.function = y
    
    def get_activation_point(self, point:float) -> float:
        if point < self.range.min(): exit("Error: Ponto fora do range")
        if point > self.range.max(): exit("Error: Ponto fora do range")
        if point <= self.a or point >= self.b: return 0
        if point == self.m: return 1

        if point > self.a and point < self.m:
            return (point-self.a) / (self.m-self.a)
        if point > self.m and point < self.b:
             return (self.b - point) / (self.b-self.m)
        