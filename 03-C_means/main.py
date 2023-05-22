from C_means import C_Means
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from utils import display_samples
from utils import display_samples_with_centers

# Definição da base de dados
n_centers = 4
test_percentage = 0.3
samples = 500
desv_pad = 1
random_seed = 3212

# Geração dos dados de amostra
x, y = make_blobs(n_samples=samples, centers=n_centers, cluster_std=desv_pad, random_state=random_seed)

# Separação dos dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_percentage, stratify=y)

# Plot das amostras
display_samples(x, y, "Conjunto das amostras", "Results/samples.png")
display_samples(x_train, y_train, "Conjunto de treinamento", "Results/train.png")
display_samples(x_test, y_test, "Conjunto de validação", "Results/test.png")

def runs(model):
    # Treinamento do modelo
    model.fit(x=x_train, y=y_train, n_centers=n_centers)
    model.info()

    # Plot dos clusters em cada época
    for epoch in range(len(model.historic['centers']) - 1):
        display_samples_with_centers(
            input=x_train,
            centers=model.historic['centers'][epoch],
            membership_matrix=model.historic['membership_matrix'][epoch],
            title=f"Época {epoch}",
            show=False,
            save_path="Results/" + ("supervised" if model.supervised else "unsupervised") + "/train"
        )

    # Predição do conjunto de teste e plot dos resultados
    Y, erro, accuracy = model.predict_group(x_test, y_test)
    display_samples_with_centers(
        input=x_test,
        membership_matrix=Y,
        centers=model.centers,
        title="Resultado conjunto de testes",
        show=False,
        save_path="Results/" + ("supervised" if model.supervised else "unsupervised") + "/test"
    )

    # Exibição dos resultados
    print("\nResultados obtidos: ")
    print("\tNúmero de erros:", erro)
    print(f"\tPorcentagem de acerto: {round(accuracy * 100, 2)}%")

# Criação do modelo não supervisionado
model_unsupervised = C_Means(supervised=False)

# Criação do modelo supervisionado
model_supervised = C_Means(supervised=True, centers=[])

# Execução do modelo não supervisionado
runs(model_unsupervised)

# Execução do modelo supervisionado
runs(model_supervised)
