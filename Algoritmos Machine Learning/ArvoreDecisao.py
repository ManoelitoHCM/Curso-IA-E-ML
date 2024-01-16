import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

#importando os dados da planilha e eliminando 1a coluna com dados irrelevantes
base = pd.read_csv('insurance.csv')
base = base.drop(columns=['Unnamed: 0'])

print(base)

#atribuição da coluna que será prevista a y (variável dependente)
y = base.iloc[:, 7].values

#atribuição das colunas que serão usadas na predição a x (variáveis independentes)
x = base.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]].values

labelencoder = LabelEncoder()

#percorre todas as colunas de x e as transforma em número caso não sejam objetos
for i in range(x.shape[1]):
    if x[:, i].dtype == 'object':
        x[:, i] = labelencoder.fit_transform(x[:, i])

#transformando y em numero
y = labelencoder.fit_transform(y)

#divisão das variaveis teste e treinamento:
#treinamento recebe 70% das variaveis independentes e treino recebe 30%
#random_state igual garante que a divisão aleatória será mantida em outras execuções
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.3, random_state=0)

#criação do modelo
modelo = GaussianNB()

#atribuição das colunas de treinamento ao modelo
modelo.fit(x_treinamento, y_treinamento)

previsao = modelo.predict(x_teste)

#calculo das metricas de avaliação relacionando o y teste e o y previsto pelo modelo
accuracy = accuracy_score(y_teste, previsao)
precision = precision_score(y_teste, previsao, average='weighted')
recall = recall_score(y_teste, previsao, average='weighted')
f1 = f1_score(y_teste, previsao, average='weighted')

print(f'Acuracia: {accuracy}, Precisao: {precision}, Recall: {recall}, F1: {f1}')

#relatorio de classificação exibindo as métricas em detalhes
report = classification_report(y_teste, previsao)

print(report)

confusao = ConfusionMatrix(modelo, classes=['None', 'Severe', 'Mild', 'Moderate'])
confusao.fit(x_treinamento, y_treinamento)

confusao.score(x_teste, y_teste)

confusao.poof()