import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# importando os dados da planilha e eliminando a 1a coluna (possui apenas índices)
base = pd.read_csv('insurance.csv')
base = base.drop(columns=['Unnamed: 0'])

print(base)

# Logo abaixo, será feita a atribuição dos dados usando iloc. Em Python, a síntaxe [:, i] implica em percorrer todas
# as linhas (:) de uma dada coluna i.

# atribuição da coluna que será prevista a y (variável dependente)
y = base.iloc[:, 7].values

# atribuição das colunas que serão usadas na predição a x (variáveis independentes)
x = base.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]].values

labelencoder = LabelEncoder()

# Percorre todas as colunas de x e as transforma em número caso não sejam objetos
# O parâmetro shape[1] retorna o segundo índice da tupla shape(que poderia assumir valor 0 ou 1).
# 0 → linha; 1 → coluna
# dtype retorna o tipo e compara com object. Se for, vai usar label encoder, senão (se for um número), não precisará.
for i in range(x.shape[1]):
    if x[:, i].dtype == 'object':
        x[:, i] = labelencoder.fit_transform(x[:, i])

# transformando y em número
y = labelencoder.fit_transform(y)

# divisão das variáveis teste e treinamento:
# treinamento recebe 70% das variáveis independentes e teste recebe 30%
# random_state igual garante que a divisão aleatória será mantida em outras execuções
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.3, random_state=0)

# criação do modelo
modelo = GaussianNB()

# atribuição das colunas de treinamento ao modelo
modelo.fit(x_treinamento, y_treinamento)

y_predict = modelo.predict(x_teste)

# cálculo das métricas de avaliação relacionando o y teste e o y previsto pelo modelo
accuracy = accuracy_score(y_teste, y_predict)
precision = precision_score(y_teste, y_predict, average='weighted')
recall = recall_score(y_teste, y_predict, average='weighted')
f1 = f1_score(y_teste, y_predict, average='weighted')

print(f'Acuracia: {accuracy}, Precisao: {precision}, Recall: {recall}, F1: {f1}')

# relatório de classificação exibindo as métricas em detalhes
report = classification_report(y_teste, y_predict)
print(report)

# Calcula a matriz de confusão
cm = confusion_matrix(y_teste, y_predict)

# Plota a matriz de confusão usando seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['None', 'Severe', 'Mild', 'Moderate'],
            yticklabels=['None', 'Severe', 'Mild', 'Moderate'])
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()
