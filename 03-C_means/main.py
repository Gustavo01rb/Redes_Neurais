from C_means import C_Means
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Definição da base de dados
n_centers = 4
test_percentage = 0.3
samples = 700
desv_pad = 1.5
random_seed = 32121

def plot_data(input, output, title, path_to_save):
    plt.clf()        
    groups = plt.scatter(input[:,0], input[:,1], marker='o', c=output,edgecolor='k')
    plt.title(title, fontsize=20, fontweight ="bold")
    plt.grid(True)
    plt.legend(*groups.legend_elements(),
                    loc="lower left", title="Grupos")
    plt.savefig(path_to_save)


x, y = C_Means.generate_data(n_samples=samples, centers=n_centers, cluster_std=desv_pad, random_seed=random_seed)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = test_percentage, stratify=y)

#plot das amostras
plot_data(x, y, "Conjunto das amostras", "results/amostras.png")
plot_data(x_train, y_train, "Conjunto de treinamento", "results/train.png")
plot_data(x_test, y_test, "Conjunto de validação", "results/test.png")


def unsupervised():
    model = C_Means(supervised=False)
    model.fit(x_train, n_centers= n_centers)
    model.info(path="results/unsupervised_train/", x=x_train, show=False)
    Y, erro, accuracy = model.predict_group(x_test, y_test)
    plot_data(x_test, Y, "Validação não supervisionado", "results/unsupervised_test/result.png")
    print("\nResultados obtidos: ")
    print("\tNúmero de erros: ", erro)
    print(f"\tPorcentagem de acerto: {round(accuracy * 100, 2)} %")
    
def supervised():
    model = C_Means(supervised=True)
    model.fit(x=x_train,y=y_train)
    model.info(path="results/supervised_train/"  , x=x_train, show=False)
    Y, erro, accuracy = model.predict_group(x_test, y_test)
    plot_data(x_test, Y, "Validação supervisionado", "results/supervised_test/result.png")
    print("\nResultados obtidos: ")
    print("\tNúmero de erros: ", erro)
    print(f"\tPorcentagem de acerto: {round(accuracy * 100, 2)} %")

unsupervised()
#supervised()

