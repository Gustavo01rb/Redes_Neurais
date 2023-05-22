import numpy as np
import matplotlib.pyplot as plt

class C_Means:
    def __init__(
        self,
        supervised: bool = False,
        centers: list = [],
        max_epoch: int = 10,
        tol: float = 0.02
    ) -> None:
        """
        Classe que implementa o algoritmo C-Means.

        Parâmetros:
            - supervised: bool, opcional (padrão=False)
                Indica se o algoritmo é supervisionado ou não.
            - centers: list, opcional (padrão=[])
                Lista de coordenadas dos centros dos clusters.
            - max_epoch: int, opcional (padrão=10)
                Número máximo de épocas (iterações) para executar o algoritmo.
            - tol: float, opcional (padrão=0.02)
                Tolerância para a condição de parada baseada na mudança nos centros.

        Retorna:
            Nenhum.
        """
        self.tol = tol
        self.supervised = supervised
        self.centers = centers
        self.membership_matrix = []
        self.epoch = max_epoch
        self.historic = {
            "centers": [],
            "membership_matrix": []
        }

    def __euclidean_distance(self, *args):
        """
        Calcula a distância euclidiana entre vários pontos em um espaço euclidiano.

        Parâmetros:
            - args: arrays ou coordenadas dos pontos.
                Pontos para os quais a distância euclidiana será calculada.

        Retorna:
            dist: float
                Distância euclidiana entre os pontos fornecidos.
        """
        sum = 0
        for cord in range(len(args[0])):
            for point in range(len(args) - 1):
                d = args[point][cord] - args[point + 1][cord]
                sum += d ** 2
        return sum ** (1/2)

    def predict(self, point: list, centers=[]):
        """
        Retorna o índice do centro mais próximo de um ponto.

        Parâmetros:
            - point: list
                Coordenadas do ponto para o qual deseja-se encontrar o centro mais próximo.
            - centers: list, opcional (padrão=[])
                Lista de coordenadas dos centros. Se não for fornecida, utiliza os centros da classe.

        Retorna:
            index: int
                Índice do centro mais próximo do ponto.
        """
        current_center = centers if len(centers) > 0 else self.centers
        distance_to_center = [self.__euclidean_distance(point, center) for center in current_center]
        return distance_to_center.index(min(distance_to_center))

    def predict_group(self, x: np.array, y: np.array):
        """
        Realiza a predição dos grupos (rótulos) para um conjunto de amostras e compara com os rótulos esperados.

        Parâmetros:
            - x: np.array
                Conjunto de amostras a serem classificadas.
            - y: np.array
                Rótulos esperados para as amostras.

        Retorna:
            Y: list
                Rótulos preditos para as amostras.
            errors: int
                Número de amostras classificadas incorretamente.
            accuracy: float
                Acurácia da classificação (proporção de amostras classificadas corretamente).
        """
        Y = []
        errors = 0
        for index, point in enumerate(x):
            out = self.predict(point)
            Y.append(out)
            if out != y[index]:
                errors += 1
        return Y, errors, (y.shape[0] - errors) / y.shape[0]

    def __init_random_center(self, n_centers, x, tol=2) -> None:
        """
        Inicializa os centros de forma aleatória para o caso não supervisionado.

        Parâmetros:
            - n_centers: int
                Número de centros a serem gerados.
            - x: np.array
                Conjunto de amostras.
            - tol: float, opcional (padrão=2)
                Tolerância para a geração aleatória dos centros.

        Retorna:
            Nenhum.
        """
        if len(self.centers) != 0:
            return
        x1_min = x[:, 0].min()
        x1_max = x[:, 0].max()
        x2_min = x[:, 1].min()
        x2_max = x[:, 1].max()
        for _ in range(n_centers):
            x_cord = np.random.uniform(x1_min - tol, x1_max + tol)
            y_cord = np.random.uniform(x2_min - tol, x2_max + tol)
            self.centers.append(np.array([x_cord, y_cord]))
        self.centers = np.array(self.centers)

    def __init_supervised_centers(self, x, y) -> None:
        """
        Inicializa os centros com base nos rótulos fornecidos para o caso supervisionado.

        Parâmetros:
            - x: np.array
                Conjunto de amostras.
            - y: np.array
                Rótulos das amostras.

        Retorna:
            Nenhum.
        """
        for center in np.unique(y):
            x_cord = x[y==center][:,0].mean()
            y_cord = x[y==center][:,1].mean()
            new_center = np.array([x_cord, y_cord])
            self.centers.append(new_center)

        self.centers = np.array(self.centers)

    def __accuracy(self, x, y, centers):
        """
        Calcula a acurácia da classificação com base nos centros fornecidos.

        Parâmetros:
            - x: np.array
                Conjunto de amostras.
            - y: np.array
                Rótulos das amostras.
            - centers: np.array
                Centros utilizados na classificação.

        Retorna:
            accuracy: float
                Acurácia da classificação (proporção de amostras classificadas corretamente).
        """
        errors = 0
        for index, expected_y in enumerate(y):
            output = self.predict(x[index, :], centers)
            if output != expected_y:
                errors += 1
        return (y.shape[0] - errors) / y.shape[0]

    def fit(self, x: np.array, y=np.array, n_centers: int = None) -> None:
        """
        Executa o treinamento do modelo de acordo com os dados fornecidos.

        Parâmetros:
            - x: np.array
                Conjunto de amostras.
            - y: np.array, opcional (padrão=None)
                Rótulos das amostras. Opcionalmente utilizado no caso supervisionado.
            - n_centers: int, opcional (padrão=None)
                Número de centros a serem gerados. Utilizado no caso não supervisionado.

        Retorna:
            Nenhum.
        """
        if self.supervised:
            if len(y) == 0:
                print("[ERRO] -> Conjunto de saída não informado")
                exit()
            self.__init_supervised_centers(x, y)
            self.__fit_supervised(x, y)
            return
        if n_centers is None:
            print("[ERRO] -> Número de centros não informado.")
            exit()
        self.__init_random_center(n_centers=n_centers, x=x)
        self.__fit_unsupervised(x)

    def __fit_supervised(self, x, y):
        """
        Executa o treinamento no caso supervisionado.

        Parâmetros:
            - x: np.array
                Conjunto de amostras.
            - y: np.array
                Rótulos das amostras.

        Retorna:
            Nenhum.
        """
        self.historic['centers'].append(self.centers.copy())
        for current_epoch in range(self.epoch):
            self.membership_matrix = np.array([self.predict(point) for point in x])
            self.historic['membership_matrix'].append(self.membership_matrix.copy())

            bcentroid = []
            for center, _ in enumerate(self.centers):
                points = self.membership_matrix == center
                if not True in points:
                    continue
                bcentroid.append(np.array([
                    x[points, 0].mean(),
                    x[points, 1].mean()
                ]))
            bcentroid = np.array(bcentroid)

            accuracy_centroid = self.__accuracy(x, y, self.centers)
            accuracy_bcentroid = self.__accuracy(x, y, bcentroid)
            if accuracy_bcentroid > accuracy_centroid:
                self.centers = bcentroid
                self.historic['centers'].append(self.centers.copy())
            else:
                self.epoch = current_epoch + 1
                return

    def __fit_unsupervised(self, x):
        """
        Executa o treinamento no caso não supervisionado.

        Parâmetros:
            - x: np.array
                Conjunto de amostras.

        Retorna:
            Nenhum.
        """
        self.historic['centers'].append(self.centers.copy())
        for current_epoch in range(self.epoch):
            self.membership_matrix = np.array([self.predict(point) for point in x])
            self.historic['membership_matrix'].append(self.membership_matrix.copy())

            for center, _ in enumerate(self.centers):
                points = self.membership_matrix == center
                if not True in points:
                    continue
                self.centers[center] = np.array([
                    x[points, 0].mean(),
                    x[points, 1].mean()
                ])

            change = np.max(np.abs(self.centers - self.historic['centers'][-1]))
            if change < self.tol:
                self.epoch = current_epoch + 1
                return
            self.historic['centers'].append(self.centers.copy())
    
    def info(self) -> None:
        """
        Exibe informações sobre o modelo, como o número de épocas até a convergência e a posição final dos centros.
        Também chama a função show() para exibir visualizações dos dados.

        Parâmetros:
        - path (str): Caminho para salvar as visualizações.
        - x (np.array): Array de entrada com as amostras.
        - show (bool, opcional): Indica se as visualizações devem ser exibidas na tela. O padrão é False.
        """
        print("\n\n Informações do modelo: ")
        print("\t Número de épocas até a convergência: ", self.epoch)
        print("\tPosição final dos centros: ")
        for i, center in enumerate(self.centers):
            print(f"\t\tCentro {i+1} : {center}")
        

