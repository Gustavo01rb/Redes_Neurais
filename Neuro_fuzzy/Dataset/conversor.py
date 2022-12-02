import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def data_reader(test_percent : float) -> np.ndarray:
    x = pd.read_csv('Dataset/xt.xls', header=None, sep=';')
    x = np.array(x)
    y = pd.read_csv('Dataset/yt.xls', header=None, sep=';')
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = test_percent)
    return x, y, x_train, x_test, y_train, y_test  