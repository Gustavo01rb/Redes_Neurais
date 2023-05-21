import numpy as np
import pandas as pd

def data_reader(test_percent : float) -> np.ndarray:
    x = pd.read_csv('Dataset/xt.xls', header=None, sep=';')
    x = np.array(x)
    y = pd.read_csv('Dataset/yt.xls', header=None, sep=';')
    y = np.array(y)
    train_size  = int(y.shape[0] * (1 - test_percent))
    print(f"Total de amostras : {y.shape[0]} ")
    print(f"Tamanho do conjunto de treinamento : {train_size} ")
    print(f"Tamanho do conjunto de validação : {y.shape[0] - train_size} ")
    x_train = x[0:train_size , :]
    x_test  = x[train_size ::, :]
    y_train = y[0:train_size , :]
    y_test  = y[train_size ::, :]
    return x, y, x_train, x_test, y_train, y_test  