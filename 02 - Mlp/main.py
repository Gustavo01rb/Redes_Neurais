from Mlp import MLP
import numpy as np
import sys
sys.path.append('..')
from utils.Graphs import Graphs

def gate_test(X,Y, title):
    mlp = MLP(dims=[2, 5, 1], eta=0.1, activation='sigmoid', max_epochs=4000, alpha=0.55)
    mlp.fit(X, Y)
    Graphs.display_mlp(mlp, X,Y, title, save_path=f"Results/{title}/", show=False, erro=True)
    

# Porta lógica AND
X_and = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

Y_and = np.array([[0],[0],[0],[1]])
gate_test(X_and, Y_and, "and_gate")

# Porta lógica OR
X_or = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])
Y_or = np.array([[0],[1],[1],[1]])
gate_test(X_or, Y_or, "or_gate")

# Porta lógica XOR
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
Y_xor = np.array([[0],[1],[1],[0]])
gate_test(X_xor, Y_xor, "xor_gate")


