import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

mtcars = pd.read_csv('mt_cars.csv')

print(mtcars)

# atribuição das colunas e divisão de variáveis dependentes e independentes
x = mtcars[['mpg', 'hp']].values
y = mtcars['cyl'].values

# divisão das variáveis teste e treinamento:
# treinamento recebe 70% das variáveis independentes e teste recebe 30%
# random_state igual garante que a divisão aleatória será mantida em outras execuções
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.3, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3)

modelo = knn.fit(x_treinamento, y_treinamento)

y_predict = modelo.predict(x_teste)

# cálculo das métricas de avaliação relacionando o y teste e o y previsto pelo modelo
accuracy = accuracy_score(y_teste, y_predict)
precision = precision_score(y_teste, y_predict, average='weighted')
recall = recall_score(y_teste, y_predict, average='weighted')
f1 = f1_score(y_teste, y_predict, average='weighted')

print(f'Acuracia: {accuracy}, Precisao: {precision}, Recall: {recall}, F1: {f1}')

# Calcula a matriz de confusão
cm = confusion_matrix(y_teste, y_predict)

print('Matriz de confusão:\n', cm)

# Plota a matriz de confusão usando seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['4', '6', '8'], yticklabels=['4', '6', '8'])
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()
