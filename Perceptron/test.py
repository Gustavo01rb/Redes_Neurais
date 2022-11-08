import sys
import pandas as pd
import numpy as np
sys.path.append('..')
from utils.Graphs import Graphs
import matplotlib.pyplot as plt
from Perceptron import Perceptron
from sklearn.datasets import make_classification


def sklearn_test():
    x, y = make_classification(n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1)

    model =  Perceptron()
    model.fit(x,y)

    Graphs.plot_2d_graph(
        x,
        y,
        show=False, 
        x_axis=False, 
        y_axis=False, 
        model_weights=model.weights, 
        history_weights=model.history['weights'],
        history_errors=model.history['errors'],
        save_path="Results/Sklearn_test/",
        title="Sklearn_datasets")
    model.show_info("Sklearn.datasets")

def or_gate_test():
    or_truth_table = [
        [0,0,0],
        [0,1,1],
        [1,0,1],
        [1,1,1]
    ]   
    or_truth_table = pd.DataFrame(or_truth_table, columns=['X1','X2','Y'])
    x = np.array([or_truth_table['X1'], or_truth_table['X2']]).T
    y = or_truth_table['Y'] 

    model =  Perceptron()
    model.fit(x,y)

    Graphs.plot_2d_graph(
        x,
        y,
        show=False, 
        x_axis=False, 
        y_axis=False, 
        model_weights=model.weights, 
        history_weights=model.history['weights'],
        history_errors=model.history['errors'],
        save_path="Results/or_gate/",
        title="or_gate")
    model.show_info("or_gate")

def and_gate_test():
    and_truth_table = [
        [0,0,0],
        [0,1,0],
        [1,0,0],
        [1,1,1]
    ]   
    and_truth_table = pd.DataFrame(and_truth_table, columns=['X1','X2','Y'])
    x = np.array([and_truth_table['X1'], and_truth_table['X2']]).T
    y = and_truth_table['Y'] 

    model =  Perceptron()
    model.fit(x,y)

    Graphs.plot_2d_graph(
        x,
        y,
        show=False, 
        x_axis=False, 
        y_axis=False, 
        model_weights=model.weights, 
        history_weights=model.history['weights'],
        history_errors=model.history['errors'],
        save_path="Results/and_gate/",
        title="and_gate")
    model.show_info("and_gate")

sklearn_test()
or_gate_test()
and_gate_test()