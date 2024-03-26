from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import confusion_matrix


def plot_cluster(data, labels, title):
    colors = ['red', 'green', 'purple', 'black']
    plt.subplots(figsize=(8, 4))

    # i vai ser range (-1 a 3), c vai ser para as cores e
    # l será lista com nome das classes
    for i, c, l in zip(range(-1, 3), colors, ['Noise', 'Setosa', 'Versicolor', 'Virginica']):
        # para os casos sem cluster definido (serão marcados com 'x')
        if i == -1:
            plt.scatter(data[labels == i, 0], data[labels == i, 3], c=colors[i], label=l, alpha=0.5, s=50, marker='x')
        # para os demais casos
        else:
            plt.scatter(data[labels == i, 0], data[labels == i, 3], c=colors[i], label=l, alpha=0.5, s=50)
    plt.legend()
    plt.title(title)
    plt.xlabel('Comprimento da sépala')
    plt.ylabel('Largura da pétala')

    plt.show()


iris = datasets.load_iris()
# print(iris)

kmeans = KMeans(n_clusters=3, n_init='auto')
kmeans.fit(iris.data)

result_kmeans = confusion_matrix(iris.target, kmeans.labels_)
print(result_kmeans)

plot_cluster(iris.data, kmeans.labels_, 'Cluster KMeans')

dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan_labels = dbscan.fit_predict(iris.data)

plot_cluster(iris.data, dbscan_labels, 'Cluster DBSCAN')

agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(iris.data)

plot_cluster(iris.data, agglo_labels, 'Cluster Hierarquico')

result_agglo = confusion_matrix(iris.target, agglo.labels_)
print(result_agglo)

plt.figure(figsize=(12, 6))
plt.title("Cluster Hierarquico: Dendograma")
plt.xlabel('Indice')
plt.ylabel('Distancia')

linkage_matrix = linkage(iris.data, method='ward')
dendrogram(linkage_matrix, truncate_mode='lastp', p=15)
plt.axhline(y=7, c='gray', lw=1, linestyle='dashed')
plt.show()
