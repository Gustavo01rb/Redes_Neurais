import sys
import pandas as pd
import numpy as np
sys.path.append('..')
from utils.Graphs import Graphs
from Perceptron import Perceptron
from sklearn.datasets import make_classification

def sklearn_test():
    x, y = make_classification(n_features=2, n_redundant=0, n_informative=1,
                               n_clusters_per_class=1)

    model = Perceptron()
    model.fit(x, y)

    save_path = "Results/Sklearn_test/"
    title = "Sklearn_datasets"
    Graphs.plot_2d_graph(
        x,
        y,
        show=False, 
        x_axis=False, 
        y_axis=False, 
        model_weights=model.weights, 
        history_weights=model.history['weights'],
        history_errors=model.history['errors'],
        save_path=save_path,
        title=title
    )
    """
        Testa o Perceptron usando o conjunto de dados do `make_classification` do scikit-learn.
    """

def gate_test(gate_name, gate_data):
    gate_table = pd.DataFrame(gate_data, columns=['X1', 'X2', 'Y'])
    x = np.array([gate_table['X1'], gate_table['X2']]).T
    y = gate_table['Y']

    model = Perceptron()
    model.fit(x, y)

    save_path = f"Results/{gate_name}/"
    title = gate_name
    Graphs.plot_2d_graph(
        x,
        y,
        show=False, 
        x_axis=False, 
        y_axis=False, 
        model_weights=model.weights, 
        history_weights=model.history['weights'],
        history_errors=model.history['errors'],
        save_path=save_path,
        title=title)
    model.show_info(title)
    """
        Testa o Perceptron usando um conjunto de dados de uma porta lógica específica.

        Parâmetros:
        - gate_name: Nome da porta lógica.
        - gate_data: Lista contendo os dados da tabela da verdade da porta lógica.
    """

def or_gate_test():
    or_gate_data = [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    gate_test("or_gate", or_gate_data)
    """
        Testa o Perceptron usando a porta lógica OR.
    """

def and_gate_test():
    and_gate_data = [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1]
    ]
    gate_test("and_gate", and_gate_data)
    """
        Testa o Perceptron usando a porta lógica AND.
    """

def xor_gate_test():
    xor_gate_data = [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]
    gate_test("xor_gate", xor_gate_data)
    """
        Testa o Perceptron usando a porta lógica XOR.
    """


#Executando os testes:
sklearn_test()
and_gate_test()
or_gate_test()
xor_gate_test()