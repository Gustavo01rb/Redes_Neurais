import Dataset.conversor as conversor
import numpy as np     
from NFN import NFN
from helpers.graphs import Graphs
from helpers.triangle_mf import Triangle_MF as TMF

x, y, x_train, x_test, y_train, y_test = conversor.data_reader(0.2)

# Plot da saída
range_y = np.linspace(y.min(), y.max(), y.shape[0])
Graphs.plot_functions(
    data = y, 
    labels = "y",
    show = False, 
    range = range_y, 
    multi_functions=False, 
    title="Saída esperada", 
    save_path="images/real_output.png")


# Definição das funções de pertinência do antecedente
'''
    Como a entrada é composta por 3 entradas [X1,X2,X3] serão definidos 3 antecendentes 
    cada um com 3 funções de ativação triangulares
'''

range_x_train = np.linspace(x_train.min(), x_train.max(), x_train.shape[0])
ante = np.array([
    np.array([ #X1
        TMF(range_x_train, -2, x_train.min(), x_train.max()/2),
        TMF(range_x_train, x_train.min() , x_train.max()/2, x_train.max()),
        TMF(range_x_train, x_train.max()/2, x_train.max(),2)
    ]),
    np.array([ #X2
        TMF(range_x_train, -2, x_train.min(), x_train.max()/2),
        TMF(range_x_train, x_train.min() , x_train.max()/2, x_train.max()),
        TMF(range_x_train, x_train.max()/2, x_train.max(),2)
    ]),
    np.array([ #X3
        TMF(range_x_train, -2, x_train.min(), x_train.max()/2),
        TMF(range_x_train, x_train.min() , x_train.max()/2, x_train.max()),
        TMF(range_x_train, x_train.max()/2, x_train.max(),2)
    ])
])
# Plot dos antecedentes
for i, X in enumerate(ante):
    Graphs.plot_functions(
        range_x_train, 
        [X[j].function for j, _ in enumerate(X)], 
        multi_functions=True, 
        show= False,
        save_path=f"images/ante_X{i+1}.png",
        title=f"Antecedente X{i+1}", 
        labels=[f"X{i+1}{j+1}" for j, _ in enumerate(X)])

model = NFN()
model.fit(ante=ante, x=x_train, y= y_train)
teste, erro = model.predict(ante, x_test)
out = np.array([y_test,teste])
print("Erro médio: ", erro.mean())

range_y_test = np.linspace(teste.min(), teste.max(), teste.shape[0])
Graphs.plot_functions(
        range_y_test, 
        out, 
        multi_functions=True, 
        show= True,
        title=f"Resultado dos testes", 
        labels=["Esperado", "Obtido" ])
