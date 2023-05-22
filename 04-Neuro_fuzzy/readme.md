# Neuro Fuzzy

Este repositório contém a implementação de um sistema Neuro-Fuzzy, que combina redes neurais e lógica fuzzy para resolver problemas de regressão. O sistema Neuro-Fuzzy é implementado usando Python e inclui funções utilitárias para pré-processamento de dados, visualização e funções de pertinência fuzzy.

### Estrutura do Projeto

* `utils.py`: Este arquivo contém funções utilitárias para leitura de dados, funções de plotagem e a definição da classe de função de pertinência triangular.
* `nfn.py`: Este arquivo implementa a classe do sistema Neuro-Fuzzy, que inclui os métodos de passagem direta e inversa para treinar a rede.
* `main.py`: Este arquivo demonstra o uso do sistema Neuro-Fuzzy realizando o pré-processamento dos dados, definindo antecedentes e funções de pertinência, treinando o modelo e fazendo previsões.

### Dependências
* NumPy: para operações numéricas e manipulação de arrays.
* Pandas: para leitura e manipulação de dados em arquivos CSV.
* Matplotlib: para visualização de dados e plotagem.

### Como Acessar

1. Clone este repositório: git clone https://github.com/Gustavo01rb/Redes_Neurais.git
2. Navegue até o diretório clonado: cd 04-Neuro_fuzzy/
3. Instale as dependências:
    * matplotlib->3.5.2
    * numpy->1.21.5
    * pandas->1.4.4

~~~
pip install -r requirements.txt 
~~~
>Comando para instalar as dependências
4. Execute o script `main.py` para executar os testes.

### Uso

1. Importe os módulos e funções necessários do repositório:
    ```python
    from nfn import NFN
    from utils import plot_functions, Triangle_MF, data_reader
    ```
2. Leia os dados de entrada e saída usando a função data_reader:
    ```python
    x, y, x_train, x_test, y_train, y_test = data_reader(test_percent=0.2)
    ```

3. Defina os antecedentes e suas funções de pertinência correspondentes. Cada antecedente é representado por uma lista de objetos Triangle_MF:
    ```python
    ante = [
        [  # Antecedente 1
            Triangle_MF(-2, x_train.min(), x_train.max() / 2),
            Triangle_MF(x_train.min(), x_train.max() / 2, x_train.max()),
            Triangle_MF(x_train.max() / 2, x_train.max(), 2)
        ],
        [  # Antecedente 2
            Triangle_MF(-2, x_train.min(), x_train.max() / 2),
            Triangle_MF(x_train.min(), x_train.max() / 2, x_train.max()),
            Triangle_MF(x_train.max() / 2, x_train.max(), 2)
        ],
        [  # Antecedente 3
            Triangle_MF(-2, x_train.min(), x_train.max() / 2),
            Triangle_MF(x_train.min(), x_train.max() / 2, x_train.max()),
            Triangle_MF(x_train.max() / 2, x_train.max(), 2)
        ]
    ]
    ```
4. Crie uma instância do sistema Neuro-Fuzzy e treine o modelo:
    ```python
    model = NFN(fixed_alpha=True, alpha=0.5, epoch=1)
    model.fit(ante=ante, x=x_train, y=y_train)
    ```

5. Faça previsões usando o modelo treinado:
    ```python
    predicted_x_train, erro_train = model.predict(ante, x_train)
    ```

6. Visualize os resultados usando a função plot_functions do arquivo utils.py:
    ```python
        plot_functions(
        x_train,
        predicted_x_train,
        multi_functions=False,
        show=True,
        title="Validação com o conjunto de treinamento",
        labels="y_p"
    )
    ```

### Resultado

<a href = "images/results/x_test_c.png">
    <img src="images/results/x_test_c.png" alt="Resultado" width="900">
</a>

### Contribuindo
Contribuições são bem-vindas! Se você encontrar algum problema, tiver ideias de melhorias ou quiser adicionar novos recursos, fique à vontade para abrir uma issue ou enviar um pull request.

### Licença
Este projeto está licenciado sob a [MIT License](../LICENSE).