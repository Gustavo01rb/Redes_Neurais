import numpy as np
import matplotlib.pyplot as plt

class Graphs:
    @staticmethod
    def plot_2d_graph(input, output, x_axis=True, y_axis=True, grid=True, show=False, save_path=None,
                      model_weights=None, title="", history_weights=None, history_errors=None):
        """
            Plota um gráfico 2D com pontos de entrada e saída, além de opcionalmente exibir o modelo de pesos,
            histórico de pesos e histórico de erros.

            Argumentos:
            - input: array numpy de forma (n, 2) com as coordenadas x e y dos pontos de entrada.
            - output: array numpy de forma (n,) com os valores de saída correspondentes aos pontos de entrada.
            - x_axis: bool (padrão True), se True exibe o eixo x.
            - y_axis: bool (padrão True), se True exibe o eixo y.
            - grid: bool (padrão True), se True exibe a grade do gráfico.
            - show: bool (padrão False), se True exibe o gráfico após a plotagem.
            - save_path: string (padrão None), o caminho do diretório para salvar as imagens do gráfico.
            - model_weights: array numpy de forma (3,) (padrão None), os pesos do modelo para traçar a reta.
            - title: string (padrão ""), o título do gráfico.
            - history_weights: lista de arrays numpy de forma (3,) (padrão None), os pesos do histórico do modelo.
            - history_errors: lista de floats (padrão None), os erros históricos do modelo.

            Retorno:
            Nenhum.
        """
        plt.clf()
        _, ax = plt.subplots(figsize=(11, 6))
        
        ax.scatter(input[:, 0], input[:, 1], marker='o', c=output, edgecolor='k')
        
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title, fontsize=22, fontweight="bold")

        if model_weights is not None and len(model_weights) > 0:
            X = np.linspace(xmin, xmax, 50)
            Y = (-X * model_weights[1] + model_weights[0]) / model_weights[2]
            ax.plot(X, Y)
        
       
        ax.grid(grid)
       
        if x_axis:
            ax.axvline(0, -1, 1, color='k', linewidth=1)
        if y_axis:
            ax.axhline(0, -2, 4, color='k', linewidth=1)
        
        if save_path:
            plt.savefig(save_path + title + '_result.png')
        
        if history_weights is not None and len(history_weights) > 0:
            _, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(title + '_history', fontsize=22, fontweight="bold")
            ax.scatter(input[:, 0], input[:, 1], marker='o', c=output, edgecolor='k')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        
           
            ax.grid(grid)
           
            if x_axis:
                ax.axvline(0, -1, 1, color='k', linewidth=1)
            if y_axis:
                ax.axhline(0, -2, 4, color='k', linewidth=1)
            
            X = np.linspace(xmin, xmax, 50)
            for index, weights in enumerate(history_weights):
                Y = (-X * weights[1] + weights[0]) / weights[2]
                ax.plot(X, Y, label="Tentativa: " + str(index))
            ax.legend(prop={'size': 12})
            if save_path:
                plt.savefig(save_path + title + '_history.png')
        
        if show:
            plt.show()
        
        if history_errors is not None and len(history_errors) > 0:
            _, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(title + '_Errors', fontsize=22, fontweight="bold")
            ax.plot(history_errors, "-r")
            
            plt.grid(grid)
            
            if save_path:
                plt.savefig(save_path + title + '_errors.png')
            if show:
                plt.show()
