import graphviz
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# importando os dados da planilha e eliminando 1.ª coluna com dados irrelevantes
base = pd.read_csv(
    r'C:\Users\manoe\OneDrive\Documentos\Workspace\Curso IA E ML\Algoritmos Machine Learning\insurance.csv')
base = base.drop(columns=['Unnamed: 0'])

print(base)

# atribuição da coluna que será prevista a y (variável dependente)
y = base.iloc[:, 7].values

# atribuição das colunas que serão usadas na predição a x (variáveis independentes)
x = base.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]].values

labelencoder = LabelEncoder()

# percorre todas as colunas de x e as transforma em número caso não sejam objetos
for i in range(x.shape[1]):
    if x[:, i].dtype == 'object':
        x[:, i] = labelencoder.fit_transform(x[:, i])

# transformando y em numero
y = labelencoder.fit_transform(y)

# divisão das variáveis teste e treinamento:
# treinamento recebe 70% das variaveis independentes e treino recebe 30%
# random_state igual garante que a divisão aleatória será mantida em outras execuções
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.3, random_state=0)

# criação do modelo
modelo = DecisionTreeClassifier(random_state=1, max_depth=6, max_leaf_nodes=10)

# atribuição das colunas de treinamento ao modelo
modelo.fit(x_treinamento, y_treinamento)

y_predict = modelo.predict(x_teste)

# calculo das métricas de avaliação relacionando o y teste e o y previsto pelo modelo
accuracy = accuracy_score(y_teste, y_predict)
precision = precision_score(y_teste, y_predict, average='weighted')
recall = recall_score(y_teste, y_predict, average='weighted')
f1 = f1_score(y_teste, y_predict, average='weighted')

print(f'Acurácia: {accuracy}, Precisão: {precision}, Recall: {recall}, F1: {f1}')

dot_data = export_graphviz(modelo, out_file=None, filled=True, feature_names=base.columns[:-1], class_names=True,
                           rounded=True)

graph = graphviz.Source(dot_data)
graph.render("Decision_tree", format="png")

# relatorio de classificação exibindo as métricas em detalhes
# report = classification_report(y_teste, previsao)

# print(report)
