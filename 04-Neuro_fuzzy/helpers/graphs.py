import matplotlib.pyplot as plt
import numpy as np

class Graphs:
    @staticmethod
    def plot_functions(
        range  : np.linspace, 
        data   : np.ndarray, 
        labels : list,
        title  : str,
        grid   : bool = True,
        show   : bool = True,
        save_path : str = None,
        multi_functions : bool = False) -> None:
        
        plt.clf()
        figure = plt.figure()
        figure.set_figwidth(10)
        plt.subplots_adjust(bottom=0.2)
        plt.title(title, fontsize=18, fontweight ="bold")
        plt.grid(grid)
        if not multi_functions:
            plt.plot(range, data, label = labels)
        else:
            for index, function in enumerate(data):
                plt.plot(range, function, label = labels[index])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=8, prop={'size': 10})
        if save_path != None: plt.savefig(save_path)
        if show : plt.show()
        plt.clf()
