import numpy as np
import matplotlib.pyplot as plt

class Graphs:
    @staticmethod
    def plot_2d_graph(input, output, x_axis = True, y_axis = True, grid = True, show = False, save_path = None, model_weights = list(), title= "", history_weights = list(), history_errors = list()):
        plt.clf()
        f = plt.figure()
        f.tight_layout()
        f.subplots_adjust(top=0.9, right=0.8)
        f.set_figwidth(10)
        
        plt.scatter(input[:, 0], input[:, 1], marker='o', c=output,edgecolor='k')
        
        xmin, xmax = plt.gca().get_xlim()
        ymin, ymax = plt.gca().get_ylim()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title,fontsize=22, fontweight ="bold")

        if len(model_weights) > 0:
            X = np.linspace(xmin,xmax,50)
            Y = (-X*model_weights[1] + model_weights[0]) / model_weights[2]
            plt.plot(X,Y)
        if grid:
            plt.grid(True)
        if x_axis:
            plt.axvline(0, -1, 1, color='k', linewidth=1)
        if y_axis:
            plt.axhline(0, -2, 4, color='k', linewidth=1)
        if save_path != None:
            plt.savefig(save_path+title+'_result.png')

        if len(history_weights) > 0:
            plt.title(title+'_history',fontsize=22, fontweight ="bold")
            X = np.linspace(xmin,xmax,50)
            for index, weights in enumerate(history_weights[:]):
                Y = (-X*weights[1] + weights[0]) / weights[2]
                plt.plot(X,Y, label="Tentativa: "+str(index))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 12})
            if save_path != None:
                plt.savefig(save_path+title+'_history.png')
        if show:
            plt.show()
        
        if len(history_errors) > 0:
            plt.clf()
            plt.title(title+'_Errors',fontsize=22, fontweight ="bold")
            plt.plot(history_errors, "-r")
            if save_path != None:
                plt.savefig(save_path+title+'_errors.png')
            if show:
                plt.show()