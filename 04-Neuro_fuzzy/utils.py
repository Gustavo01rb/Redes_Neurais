import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_reader(test_percent: float) -> np.ndarray:
    """
    Lê os dados de entrada e saída de arquivos CSV e divide-os em conjuntos de treinamento e teste.

    Args:
        test_percent (float): Percentual de dados a serem usados como conjunto de teste.

    Returns:
        tuple: Uma tupla contendo as matrizes de entrada e saída completas, bem como os conjuntos de treinamento e teste.

    Raises:
        None

    """

    x = pd.read_csv('Dataset/xt.xls', header=None, sep=';')
    x = np.array(x)
    y = pd.read_csv('Dataset/yt.xls', header=None, sep=';')
    y = np.array(y)

    train_size = int(y.shape[0] * (1 - test_percent))

    print(f"Total de amostras: {y.shape[0]}")
    print(f"Tamanho do conjunto de treinamento: {train_size}")
    print(f"Tamanho do conjunto de validação: {y.shape[0] - train_size}")

    x_train = x[0:train_size, :]
    x_test = x[train_size:, :]
    y_train = y[0:train_size, :]
    y_test = y[train_size:, :]

    return x, y, x_train, x_test, y_train, y_test

def plot_functions(
    range: np.linspace,
    data: np.ndarray,
    labels: list,
    title: str,
    grid: bool = True,
    show: bool = True,
    save_path: str = None,
    multi_functions: bool = False
) -> None:
    """
    Plota as funções fornecidas em um gráfico.

    Args:
        range (np.linspace): Array contendo os valores do eixo x.
        data (np.ndarray): Array bidimensional contendo as funções a serem plotadas.
        labels (list): Lista de rótulos para as funções.
        title (str): Título do gráfico.
        grid (bool, optional): Indica se deve exibir as linhas de grade. O padrão é True.
        show (bool, optional): Indica se deve exibir o gráfico. O padrão é True.
        save_path (str, optional): Caminho para salvar a imagem do gráfico. O padrão é None.
        multi_functions (bool, optional): Indica se são várias funções ou uma única função. O padrão é False.

    Returns:
        None

    Raises:
        None

    """

    plt.clf()
    figure = plt.figure()
    figure.set_figwidth(10)
    plt.subplots_adjust(bottom=0.2)
    plt.title(title, fontsize=18, fontweight="bold")
    plt.grid(grid)

    if not multi_functions:
        plt.plot(range, data, label=labels)
    else:
        for index, function in enumerate(data):
            plt.plot(range, function, label=labels[index])

    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
        shadow=True,
        ncol=8,
        prop={'size': 10}
    )

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.clf()

class Triangle_MF:
    def __init__(self, range: np.linspace, a: float, m: float, b: float) -> None:
        """
        Classe que representa uma função de pertinência triangular.

        Args:
            range (np.linspace): Array contendo os valores do eixo x.
            a (float): Ponto inicial do triângulo.
            m (float): Ponto médio do triângulo.
            b (float): Ponto final do triângulo.

        Returns:
            None

        Raises:
            None

        """

        self.range = range
        self.a = a
        self.m = m
        self.b = b
        y = np.zeros(range.shape[0])

        first_half = np.logical_and(a < range, range <= m)
        y[first_half] = (range[first_half] - a) / (m - a)

        second_half = np.logical_and(m <= range, range < b)
        y[second_half] = (b - range[second_half]) / (b - m)

        self.function = y

    def get_activation_point(self, point: float) -> float:
        """
        Calcula o valor de ativação para um determinado ponto de entrada.

        Args:
            point (float): Ponto de entrada.

        Returns:
            float: Valor de ativação resultante.

        Raises:
            ValueError: Se o ponto de entrada estiver fora do intervalo definido.

        """

        if point < self.range.min():
            raise ValueError("Error: Ponto fora do range")

        if point > self.range.max():
            raise ValueError("Error: Ponto fora do range")

        if point <= self.a or point >= self.b:
            return 0

        if point == self.m:
            return 1

        if point > self.a and point < self.m:
            return (point - self.a) / (self.m - self.a)

        if point > self.m and point < self.b:
            return (self.b - point) / (self.b - self.m)
