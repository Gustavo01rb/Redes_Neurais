from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class C_Means:

    def __init__(
        self,
        supervised : bool  = False,
        centers    : list  = list(),        
        max_epoch  : int   = 10,
        tol        : float = 0.02                 
    ) -> None:
        self.tol               = tol
        self.supervised        = supervised
        self.centers           = centers  
        self.membership_matrix = list()
        self.epoch             = max_epoch
        self.historic          = {
            "centers"           : list(),
            "membership_matrix" : list()
        }

    @staticmethod
    def generate_data(n_samples = 500, centers = 3, cluster_std = 0.50, random_seed = None):
        '''
            Método que retorna a base de dados para aplicação do c-means,
            a função make_blobs retorna por padrão dois valores X e Y.
            -> n_samples   = Número de amostras
            -> centers     = Número de amostras diferentes
            -> cluster_std = Desvio padrão. Interfere na distância entre os pontos
        '''    
        return  make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_seed) 

    # Métodos auxiliares
    def __euclidean_distance(self, *args):
        '''
            Essa função calcula a distânica euclidiana para N pontos.
            Os parâmetros devem ser na forma de listas: [1,2] , [3,4]
        '''
        sum = 0
        for cord in range(len(args[0])):
            for point in range(len(args) - 1):
                d = args[point][cord] - args[point + 1][cord]
                sum += d**2
        return sqrt(sum)

    def init_random_center(self, n_centers, x1_min, x1_max, x2_min, x2_max, tol = 2) -> None:
        if len(self.centers) != 0: return 
        for _ in range(n_centers):
            x_cord = np.random.uniform(x1_min - tol, x1_max + tol)
            y_cord = np.random.uniform(x2_min - tol, x2_max + tol)
            self.centers.append(np.array([x_cord, y_cord]))
        self.centers = np.array(self.centers)
    # Fim -> Métodos auxiliares


    # Métodos de ajuste e treinamento
    def fit(self, x, y = None):
        if self.supervised:
            print("Ainda não implementado")
            return
        self.fit_usupervised(x)
    
    def fit_usupervised(self, x):
        self.historic['centers'].append(self.centers.copy())
        for current_epoch in range(self.epoch):
            self.membership_matrix = list()
            for point in x:
                distance_to_center = [self.__euclidean_distance(point, center) for center in self.centers]
                self.membership_matrix.append(distance_to_center.index(min(distance_to_center)))
            self.membership_matrix = np.array(self.membership_matrix)
            self.historic['membership_matrix'].append(self.membership_matrix.copy())
            for center, _ in enumerate(self.centers):
                points = self.membership_matrix == center
                if not True in points: continue
                self.centers[center][0] = x[points,0].mean() 
                self.centers[center][1] = x[points,1].mean() 

            distance_error = np.array([self.__euclidean_distance(self.centers[i], self.historic['centers'][-1][i]) for i in range(len(self.centers))])
            self.historic['centers'].append(self.centers.copy())
            if distance_error.max() < self.tol:
                self.epoch = current_epoch + 1
                return
    # Fim -> Métodos de ajuste e treinamento
    
    # Métodos de exibição
    def info(self, x, y = None, show = False):
        print("\n\n Informações do modelo: ")
        print("\t Número de épocas até a convergência: ", self.epoch)
        print("\tPosição dos centros: ")
        for i, center in enumerate(self.centers):
            print(f"\t\tCentro {i+1} : {center}")
        self.show(x,y,show)

    def show(self, x, y, show):
        plt.clf()        
        if not self.supervised: y = self.membership_matrix
        plt.title("Amostras", fontsize=20, fontweight ="bold")
        plt.scatter(x[:,0], x[:,1], marker='o',edgecolor='k')
        plt.grid(True)
        plt.savefig("results/1_sample.png")
        if show : plt.show()

        for epoch in range(len(self.historic['centers']) -1):
            plt.clf()        
            plt.scatter(x[:,0], x[:,1], marker='o', c=y if self.supervised else self.historic['membership_matrix'][epoch],edgecolor='k')
            plt.scatter(self.historic['centers'][epoch][:,0], self.historic['centers'][epoch][:,1], marker='^',c = 'r',s=100, edgecolor='k')
            plt.title(f"Época: {epoch}", fontsize=20, fontweight ="bold")
            plt.grid(True)
            plt.savefig("results/epoch"+ str(epoch) +".png")
            if show : plt.show()    


        plt.clf()        
        plt.scatter(x[:,0], x[:,1], marker='o', c=y,edgecolor='k')
        plt.scatter(self.centers[:,0], self.centers[:,1], marker='^',c = 'r',s=100, edgecolor='k')
        plt.title("Resultado", fontsize=20, fontweight ="bold")
        plt.grid(True)
        plt.savefig("results/2_result.png")
        if show : plt.show()
