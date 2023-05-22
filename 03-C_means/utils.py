import numpy as np
import matplotlib.pyplot as plt

def display_samples_with_centers(input: np.array, centers, membership_matrix: np.array, title, show: bool = False, save_path: str = None):
    """
    Exibe os exemplos com os centros de um algoritmo de clustering.

    Parâmetros:
        - input (np.array): Array com os exemplos.
        - centers: Array com as coordenadas dos centros.
        - membership_matrix (np.array): Matriz de pertinência dos exemplos em relação aos centros.
        - title (str): Título do gráfico.
        - show (bool): Indica se o gráfico deve ser exibido.
        - save_path (str): Caminho para salvar a figura.

    Retorna:
        Nenhum.

    """
    plt.clf()
    _, ax = plt.subplots()
    groups = ax.scatter(input[:, 0], input[:, 1], marker='o', c=membership_matrix, edgecolor='k')
    ax.scatter(centers[:, 0], centers[:, 1], marker='^', c='r', label="Centros", s=100, edgecolor='k')
    plt.title(title, fontsize=20, fontweight="bold")
    ax.grid(True)
    legend2 = ax.legend(loc="upper right")
    ax.add_artist(legend2)
    legend1 = ax.legend(*groups.legend_elements(), loc="lower left", title="Grupos")
    ax.add_artist(legend1)

    if save_path is not None:
        plt.savefig(f"{save_path}/{title}.png")
    if show:
        plt.show()

def display_samples(input: np.array, output: np.array, title: str, path_to_save: str = None, show: bool = False):
    """
    Exibe os exemplos de um algoritmo de clustering.

    Parâmetros:
        - input (np.array): Array com os exemplos.
        - output (np.array): Array com as saídas do algoritmo (rótulos).
        - title (str): Título do gráfico.
        - path_to_save (str): Caminho para salvar a figura.
        - show (bool): Indica se o gráfico deve ser exibido.

    Retorna:
        Nenhum.

    """
    plt.clf()
    groups = plt.scatter(input[:, 0], input[:, 1], marker='o', c=output, edgecolor='k')
    plt.title(title, fontsize=20, fontweight="bold")
    plt.grid(True)
    plt.legend(*groups.legend_elements(), loc="lower left", title="Grupos")

    if path_to_save is not None:
        plt.savefig(path_to_save)
    if show:
        plt.show()
