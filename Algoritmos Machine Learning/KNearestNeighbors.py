import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

mtcars = pd.read_csv('mt_cars.csv')

print(mtcars)

# atribuição das colunas e divisão de variáveis dependentes e independentes
x = mtcars[['mpg', 'hp']].values
y = mtcars['cyl'].values

knn = KNeighborsClassifier(n_neighbors=3)

modelo = knn.fit(x, y)

y_predict = modelo.predict(x)

# cálculo das métricas de avaliação relacionando o y teste e o y previsto pelo modelo
accuracy = accuracy_score(y, y_predict)
precision = precision_score(y, y_predict, average='weighted')
recall = recall_score(y, y_predict, average='weighted')
f1 = f1_score(y, y_predict, average='weighted')

print(f'Acuracia: {accuracy}, Precisao: {precision}, Recall: {recall}, F1: {f1}')

# Calcula a matriz de confusão
cm = confusion_matrix(y, y_predict)

print('Matriz de confusão:\n', cm)

# 'mpg', 'hp'
new_data = np.array([[19.3, 105]])

data_prediction = modelo.predict(new_data)

# cyl
print(data_prediction)

distances, index = modelo.kneighbors(new_data)

print(distances)
print(index)

# os mais próximos que o modelo calcula para os dados de mpg e hp informados são esses 3
# e a conclusão é que cyl seria = 6 para esse caso
print(mtcars.loc[[5, 31, 1], ["Unnamed: 0", "mpg", "cyl", "hp"]])

