# Importando as bibliotecas que serão utilizadas no código.

import pandas as pd # biblioteca para manipulação dos dados
pd.set_option("display.max_columns", None) # mostrar todas as colunas do conjunto de dados
from sklearn.model_selection import train_test_split # para dividir os dados em treino e teste
from sklearn.naive_bayes import GaussianNB # para utilizar o Naive Bayes
from sklearn.preprocessing import LabelEncoder # para realizar o Label Encoder
from sklearn.metrics import accuracy_score # para calcular a acurácia do modelo
from sklearn.model_selection import cross_val_score, KFold # realizar a validação cruzada
from yellowbrick.classifier import ConfusionMatrix # para criar a matriz de confusão (em forma gráfica)

# Lendo o conjunto de dados
credito = pd.read_csv("C:/Users/55739/Desktop/Formação Cientista de Dados/FormacaoCD/31.Prática em Python/dados/Credit.csv")

# Preparação dos dados #

# Definindo quais serão as variáveis previsoras e qual variável será a prevista
previsores = credito.iloc[:,0:20].values # os previsores são os atributos de 0 a 19
classe = credito.iloc[:,20].values # a variável a ser prevista é a classe (good or bad)

# Codificação das variáveis categóricas em variáveis numéricas, usando o LabelEncoder
labelencoder1 = LabelEncoder()
previsores[:,0] = labelencoder1.fit_transform(previsores[:,0])

labelencoder2 = LabelEncoder()
previsores[:,2] = labelencoder2.fit_transform(previsores[:,2])

labelencoder3 = LabelEncoder()
previsores[:,3] = labelencoder3.fit_transform(previsores[:,3])

labelencoder4 = LabelEncoder()
previsores[:,5] = labelencoder4.fit_transform(previsores[:,5])

labelencoder5 = LabelEncoder()
previsores[:,6] = labelencoder5.fit_transform(previsores[:,6])

labelencoder6 = LabelEncoder()
previsores[:,8] = labelencoder6.fit_transform(previsores[:,8])

labelencoder7 = LabelEncoder()
previsores[:,9] = labelencoder7.fit_transform(previsores[:,9])

labelencoder8 = LabelEncoder()
previsores[:,11] = labelencoder8.fit_transform(previsores[:,11])

labelencoder9 = LabelEncoder()
previsores[:,13] = labelencoder9.fit_transform(previsores[:,13])

labelencoder10 = LabelEncoder()
previsores[:,14] = labelencoder10.fit_transform(previsores[:,14])

labelencoder11 = LabelEncoder()
previsores[:,16] = labelencoder11.fit_transform(previsores[:,16])

labelencoder12 = LabelEncoder()
previsores[:,18] = labelencoder12.fit_transform(previsores[:,18])

labelencoder13 = LabelEncoder()
previsores[:,19] = labelencoder13.fit_transform(previsores[:,19])

# Divisão da base de dados em treinamento e teste (70% para treinamento e 30% para teste)
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe, test_size=0.3, random_state=0)

# Treinamento do Modelo # 

# Criação e treinamento do modelo (geração da tabela de probabilidades)
naive_bayes = GaussianNB()
naive_bayes.fit(X_treinamento, y_treinamento) # criação do modelo

# Teste do Modelo #

# Previsões utilizando o registro de teste
previsoes = naive_bayes.predict(X_teste) # para cada registro do conjunto de teste, é feito uma previsão

# Acurácia do modelo
acuracia = accuracy_score(y_teste, previsoes)
print("Acurácia do modelo: {:.2%}".format(acuracia))

# Matriz de confusão
v = ConfusionMatrix(GaussianNB())
v.fit(X_treinamento, y_treinamento)
v.score(X_teste, y_teste)
v.poof()

# Avaliação do Modelo # 

# Validação cruzada

# Define a estratégia de validação cruzada com 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Avalia o desempenho do modelo usando validação cruzada
scores = cross_val_score(naive_bayes, previsores, classe, cv=kf, scoring='accuracy')

# Calcula a média e o desvio padrão dos resultados
print(f'\nAcurácia média: {scores.mean():.2f} +/- {scores.std():.2f}')