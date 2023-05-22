import numpy as np     
from NFN import NFN
from utils import plot_functions
from utils import Triangle_MF as TMF
from utils import data_reader


# Definição dos conjuntos
x, y, x_train, x_test, y_train, y_test = data_reader(0.2)
range_y = np.linspace(0, 1, y.shape[0])
range_y_train = np.linspace(0, 1, y_train.shape[0])
range_y_test = np.linspace(0, 1, y_test.shape[0])
range_x = np.linspace(0, 1, x.shape[0])
range_x_train = np.linspace(0, 1, x_train.shape[0])
range_x_test = np.linspace(0, 1, x_test.shape[0])

plot_functions(
    data=y,
    labels="y",
    show=False,
    range=range_y,
    multi_functions=False,
    title="Saída do sistema",
    save_path="images/Out/y.png"
)
plot_functions(
    data=y_train,
    labels="y_train",
    show=False,
    range=range_y_train,
    multi_functions=False,
    title="Saída do conjunto de treinamento",
    save_path="images/Out/y_train.png"
)
plot_functions(
    data=y_test,
    labels="y_test",
    show=False,
    range=range_y_test,
    multi_functions=False,
    title="Saída do conjunto de validação",
    save_path="images/Out/y_test.png"
)


# Definição das funções de pertinência do antecedente
'''
    Como a entrada é composta por 3 entradas [X1,X2,X3] serão definidos 3 antecendentes 
    cada um com 3 funções de ativação triangulares
'''
ante = np.array([
    np.array([  # X1
        TMF(range_x_train, -2, x_train.min(), x_train.max() / 2),
        TMF(range_x_train, x_train.min(), x_train.max() / 2, x_train.max()),
        TMF(range_x_train, x_train.max() / 2, x_train.max(), 2)
    ]),
    np.array([  # X2
        TMF(range_x_train, -2, x_train.min(), x_train.max() / 2),
        TMF(range_x_train, x_train.min(), x_train.max() / 2, x_train.max()),
        TMF(range_x_train, x_train.max() / 2, x_train.max(), 2)
    ]),
    np.array([  # X3
        TMF(range_x_train, -2, x_train.min(), x_train.max() / 2),
        TMF(range_x_train, x_train.min(), x_train.max() / 2, x_train.max()),
        TMF(range_x_train, x_train.max() / 2, x_train.max(), 2)
    ])
])

# Plot dos antecedentes
for i, X in enumerate(ante):
    plot_functions(
        range_x_train,
        [X[j].function for j, _ in enumerate(X)],
        multi_functions=True,
        show=False,
        save_path=f"images/MF/ante_X{i+1}.png",
        title=f"Antecedente X{i+1}",
        labels=[f"X{i+1}{j+1}" for j, _ in enumerate(X)]
    )

# Definição do modelo
model = NFN(fixed_alpha=True, alpha=0.5, epoch=1)
model.fit(ante=ante, x=x_train, y=y_train)

predicted_x_train, erro_train = model.predict(ante, x_train) 
comparative_x_train = np.array([y_train, predicted_x_train])

plot_functions(
    range_x_train,
    predicted_x_train,
    multi_functions=False,
    show=False,
    title=f"Validação com o conjunto de treinamento",
    save_path="images/results/x_train.png",
    labels="y_p"
)
plot_functions(
    range_x_train,
    comparative_x_train,
    multi_functions=True,
    show=False,
    save_path="images/results/x_train_c.png",
    title=f"Comparativo y_train vs y_p",
    labels=["y_train", "y_p"]
)

predicted_x_test, erro_test = model.predict(ante, x_test) 
comparative_x_test = np.array([y_test, predicted_x_test])
plot_functions(
    range_x_test,
    predicted_x_test,
    multi_functions=False,
    show=False,
    title=f"Validação com o conjunto de teste",
    save_path="images/results/x_test.png",
    labels="y_p"
)
plot_functions(
    range_x_test,
    comparative_x_test,
    multi_functions=True,
    show=False,
    save_path="images/results/x_test_c.png",
    title=f"Comparativo y_test vs y_p",
    labels=["y_test", "y_p"]
)

predicted_x, erro = model.predict(ante, x) 
comparative_x = np.array([y, predicted_x])
plot_functions(
    range_x,
    predicted_x,
    multi_functions=False,
    show=False,
    title=f"Validação com o conjunto de teste",
    save_path="images/results/x.png",
    labels="y_p"
)
plot_functions(
    range_x,
    comparative_x,
    multi_functions=True,
    show=False,
    save_path="images/results/x_c.png",
    title=f"Comparativo y vs y_p",
    labels=["y", "y_p"]
)
