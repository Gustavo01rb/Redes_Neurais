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
    
    def predict(self, point : list(), centers = list()):
        current_center = centers if len(centers) > 0 else self.centers
        distance_to_center = [self.__euclidean_distance(point, center) for center in current_center]
        return distance_to_center.index(min(distance_to_center))
    
    def predict_group(self, x : np.array, y : np.array):
        Y = list()
        errors = 0
        for index, point in enumerate(x):
            out = self.predict(point)
            Y.append(out)
            if out != y[index]:
                errors += 1
        return Y, errors, (y.shape[0] - errors) / y.shape[0]

    def __init_random_center(self, n_centers, x , tol = 2) -> None:
        if len(self.centers) != 0: return 
        x1_min = x[:,0].min()
        x1_max = x[:,0].max()
        x2_min = x[:,1].min()
        x2_max = x[:,1].max()
        for _ in range(n_centers):
            x_cord = np.random.uniform(x1_min - tol, x1_max + tol)
            y_cord = np.random.uniform(x2_min - tol, x2_max + tol)
            self.centers.append(np.array([x_cord, y_cord]))
        self.centers = np.array(self.centers)
    
    def __init_supervised_centers(self, x,y) -> None:
        for center in np.unique(y):
            group = y == center
            self.centers.append(np.array([
                x[group, 0].mean(),
                x[group, 1].mean(),
            ]))
        self.centers = np.array(self.centers)
    
    def __accuracy(self, x, y, centers):
        errors = 0
        for index, expected_y in enumerate(y):
            output = self.predict(x[index, :], centers)
            if output != expected_y :
                errors += 1
        return (y.shape[0] - errors) / y.shape[0]
        
    # Fim -> Métodos auxiliares


    # Métodos de ajuste e treinamento
    def fit(self, x : np.array, y:np.array = list, n_centers : int= None) -> None:
        if self.supervised:
            if len(y) == 0: 
                print("[ERRO] -> Conjunto de saída não informado")
                exit()
            self.__init_supervised_centers(x,y)
            self.__fit_supervised(x,y)
            return
        if n_centers == None:
            print("[ERRO] -> Número de centros não infromado.")
            exit()
        self.__init_random_center(n_centers=n_centers, x=x)
        self.__fit_usupervised(x)
    
    def __fit_supervised(self,x,y):
        self.historic['centers'].append(self.centers.copy())
        for current_epoch in range(self.epoch):
            self.membership_matrix = list()
            
            self.membership_matrix = np.array([self.predict(point) for point in x])
            self.historic['membership_matrix'].append(self.membership_matrix.copy())
            
            bcentroid = list()
            for center, _ in enumerate(self.centers):
                points = self.membership_matrix == center
                if not True in points: continue
                bcentroid.append(np.array([
                    x[points,0].mean(), 
                    x[points,1].mean()
                ])) 
            bcentroid = np.array(bcentroid)
            
            accuracy_centroid  = self.__accuracy(x,y,self.centers)
            accuracy_bcentroid = self.__accuracy(x,y,bcentroid)
            if accuracy_bcentroid > accuracy_centroid:
                self.centers = bcentroid
                self.historic['centers'].append(self.centers.copy())
            else:
                self.epoch = current_epoch+1
                return


    
    def __fit_usupervised(self, x):
        self.historic['centers'].append(self.centers.copy())
        for current_epoch in range(self.epoch):
            self.membership_matrix = list()
            
            self.membership_matrix = np.array([self.predict(point) for point in x])
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
                self.historic['centers'] = np.array(self.historic['centers'])
                self.centers = np.array(self.centers)
                return
    # Fim -> Métodos de ajuste e treinamento
    
    # Métodos de exibição
    def info(self, path : str , x : np.array, show : bool = False) -> None:
        print("\n\n Informações do modelo: ")
        print("\t Número de épocas até a convergência: ", self.epoch)
        print("\tPosição final dos centros: ")
        for i, center in enumerate(self.centers):
            print(f"\t\tCentro {i+1} : {center}")
        self.show(path, x,show)

    def show(self, path : str ,x : np.array, show : bool ) -> None:
        plt.clf()        
        y = self.membership_matrix
        plt.title("Amostras", fontsize=20, fontweight ="bold")
        plt.scatter(x[:,0], x[:,1], marker='o',edgecolor='k')
        plt.grid(True)
        plt.savefig(path + "1_sample.png")
        if show : plt.show()

        for epoch in range(len(self.historic['centers']) -1):
            plt.clf()        
            fig, ax = plt.subplots()
            groups  =  ax.scatter(x[:,0], x[:,1], marker='o', c=self.historic['membership_matrix'][epoch],edgecolor='k')
            ax.scatter(self.historic['centers'][epoch][:,0], self.historic['centers'][epoch][:,1], marker='^',c = 'r', label = "Centros",s=100,edgecolor='k')
            plt.title(f"Época: {epoch}", fontsize=20, fontweight ="bold")
            ax.grid(True)
            legend2 = ax.legend(loc="upper right")
            ax.add_artist(legend2)
            legend1 = ax.legend(*groups.legend_elements(),
                    loc="lower left", title="Grupos")
            ax.add_artist(legend1)
            

            plt.savefig(path +"epoch"+ str(epoch) +".png")
            if show : plt.show()    


        plt.clf()    
        fig, ax = plt.subplots()
        groups =  ax.scatter(x[:,0], x[:,1], marker='o', c=y,edgecolor='k')
        ax.scatter(self.centers[:,0], self.centers[:,1],label= "Centros",marker='^',c = 'r',s=100, edgecolor='k')
        plt.title("Resultado não supervisionado" if not self.supervised else "Resultado supervisionado", fontsize=20, fontweight ="bold")
        ax.grid(True)
        legend2 = ax.legend(loc="upper right")
        ax.add_artist(legend2)
        legend1 = ax.legend(*groups.legend_elements(),
                    loc="lower left", title="Grupos")
        ax.add_artist(legend1)
        plt.savefig(path +"2_result.png")
        if show : plt.show()
