import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

class Modelo():
    def __init__(self):
        self = self

    def CarregarDataset(self, path):
        """
        Carrega o conjunto de dados a partir de um arquivo CSV.

        Parâmetros:
        - path (str): Caminho para o arquivo CSV contendo o dataset.
        
        O dataset é carregado com as seguintes colunas: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm e Species.
        """
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        try:
          self.df = pd.read_csv(path, names=names)
          print("Dataset carregado com sucesso.\n")
        except FileNotFoundError:
          print("Arquivo não encontrado. Verifique o caminho especificado.\n")

    def TratamentoDeDados(self):
        """
        Realiza o pré-processamento dos dados carregados.

        Sugestões para o tratamento dos dados:
            * Utilize `self.df.head()` para visualizar as primeiras linhas e entender a estrutura.
            * Verifique a presença de valores ausentes e faça o tratamento adequado.
            * Considere remover colunas ou linhas que não são úteis para o treinamento do modelo.
        
        Dicas adicionais:
            * Explore gráficos e visualizações para obter insights sobre a distribuição dos dados.
            * Certifique-se de que os dados estão limpos e prontos para serem usados no treinamento do modelo.
        """
        # print(self.df.isnull().sum().sum()) # Sem valores NaN
        
        # Transformando os valores em rótulos de 0 a 2
        encoder = LabelEncoder()
        obj_cols = ['Species']

        values = self.df[obj_cols[0]].values
        encoder.fit(values)
        self.df[obj_cols[0]] = encoder.transform(values)
       

    def Treinamento(self):
        """
        Treina o modelo de machine learning.

        Detalhes:
            * Utilize a função `train_test_split` para dividir os dados em treinamento e teste.
            * Escolha o modelo de machine learning que queira usar. Lembrando que não precisa ser SMV e Regressão linear.
            * Experimente técnicas de validação cruzada (cross-validation) para melhorar a acurácia final.
        
        Nota: Esta função deve ser ajustada conforme o modelo escolhido.
        """
        self.X=self.df.drop('Species', axis=1)  # Features (todas as colunas exceto a de alvo)
        self.y=self.df['Species']

        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2)
        

    def Teste(self):
        """
        Avalia o desempenho do modelo treinado nos dados de teste.

        Esta função deve ser implementada para testar o modelo e calcular métricas de avaliação relevantes, 
        como acurácia, precisão, ou outras métricas apropriadas ao tipo de problema.
        """
        logreg=LogisticRegression()
        logreg.fit(self.X_train,self.y_train)

        
        y_pred=logreg.predict(self.X_test)
        print('Precisão de Regressão Logistica:')
        print(metrics.accuracy_score(self.y_test,y_pred))
        print('Score:')
        print(logreg.score(self.X, self.y))

        lreg=HistGradientBoostingClassifier()
        lreg.fit(self.X_train,self.y_train)

        
        y_pred=lreg.predict(self.X_test)
        print('Precisão de Gradiente Booster:')
        print(metrics.accuracy_score(self.y_test,y_pred))
        print('Score:')
        print(lreg.score(self.X, self.y))

        knn=KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.X_train,self.y_train)
        
        y_pred=knn.predict(self.X_test)
        print('Precisao do KNN com 5 vizinhos')
        print(metrics.accuracy_score(self.y_test,y_pred))
        print('Score:')
        print(knn.score(self.X, self.y))

    def Train(self):
        """
        Função principal para o fluxo de treinamento do modelo.

        Este método encapsula as etapas de carregamento de dados, pré-processamento e treinamento do modelo.
        Sua tarefa é garantir que os métodos `CarregarDataset`, `TratamentoDeDados` e `Treinamento` estejam implementados corretamente.
        
        Notas:
            * O dataset padrão é "iris.data", mas o caminho pode ser ajustado.
            * Caso esteja executando fora do Colab e enfrente problemas com o path, use a biblioteca `os` para gerenciar caminhos de arquivos.
        """
        self.CarregarDataset("iris.data")  # Carrega o dataset especificado.

        # Tratamento de dados opcional, pode ser comentado se não for necessário
        self.TratamentoDeDados()

        self.Treinamento()  # Executa o treinamento do modelo

        self.Teste()

# Lembre-se de instanciar as classes após definir suas funcionalidades
# Recomenda-se criar ao menos dois modelos (e.g., Regressão Linear e SVM) para comparar o desempenho.
# A biblioteca já importa LinearRegression e SVC, mas outras escolhas de modelo são permitidas.
