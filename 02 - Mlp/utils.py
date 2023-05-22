import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction:
    @staticmethod
    def step_function_1(x):
        return np.where(x >= 0, 1, 0)

    @staticmethod
    def step_function_2(x):
        return np.where(x >= 0, 1, -1)

    @staticmethod
    def step_function_3(x):
        return np.where(x > 0, 1, np.where(x == 0, 0, -1))

    @staticmethod
    def linear_function(x):
        return np.where(x > 1, 1, np.where(x < 0, 0, x))

    @staticmethod
    def linear_function_no_saturation(x, a):
        return a * x

    @staticmethod
    def sigmoidal_function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoidal_function(x):
        sigmoid = ActivationFunction.sigmoidal_function(x)
        return sigmoid * (1 - sigmoid)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def dtanh(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def drelu(x):
        return np.where(x > 0, 1, 0)

def display_mlp(mlp, X, Y, title, show=False, erro=True, save_path=None):
    """
    Exibe a MLP treinada e os pontos de dados no plano XY.

    Args:
    - mlp (MLP): Objeto MLP treinado.
    - X (numpy.ndarray): Matriz de entrada.
    - Y (numpy.ndarray): Matriz de saída.
    - title (str): Título do gráfico.
    - show (bool): Indica se o gráfico deve ser exibido (padrão: False).
    - erro (bool): Indica se o gráfico de erros deve ser exibido (padrão: True).
    - save_path (str): Caminho para salvar os gráficos (padrão: None).
    """

    plt.clf()
    X_scatter = X.copy()
    Y_scatter = Y.copy()

    X_grid = np.linspace(-0.5, 1.5, 100)
    Y_grid = np.linspace(-0.5, 1.5, 100)
    X_grid, Y_grid = np.meshgrid(X_grid, Y_grid)

    def F(x, y):
        return mlp.predict(np.array([[x, y]]))

    Z = np.vectorize(F)(X_grid, Y_grid)
    plt.pcolor(X_grid, Y_grid, Z, cmap='jet')
    plt.colorbar()
    cntr = plt.contour(X_grid, Y_grid, Z, levels=[0.5])

    colors = ['g' if y[0] == 1 else 'y' for y in Y_scatter]
    plt.scatter(X_scatter[:, 0], X_scatter[:, 1], s=300, marker='o', c=colors, edgecolors='k')

    plt.clabel(cntr, inline=1, fontsize=10)

    plt.grid()
    plt.title(title, fontsize=22, fontweight="bold")

    if save_path is not None:
        plt.savefig(save_path + title + '_result.png')

    if show:
        plt.show()

    if erro:
        plt.clf()
        plt.grid()
        plt.plot(mlp.train_error, "-r")
        plt.title(title + " errors", fontsize=22, fontweight="bold")

        if save_path is not None:
            plt.savefig(save_path + title + '_errors.png')

        if show:
            plt.show()