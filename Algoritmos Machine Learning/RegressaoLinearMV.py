import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns

#Esse script faz alguns testes de regressão linear com mais de uma variável dependente
#Lê o arquivo usado como base de dados
base = pd.read_csv('mt_cars.csv')

#Elimina a primeira coluna que contém os nomes por ser irrelevante
base = base.drop(['Unnamed: 0'], axis = 1)

print(base.head())

#Computa correlação entre as colunas
corr = base.corr()

#Plota um mapa de calor para tornar mais evidente as correlações mais fortes
sns.heatmap(corr, cmap = 'coolwarm', annot = True, fmt = '.2f')

#Definição dos pares de coluna x e y para os gráficos
column_pairs = [('mpg','cyl'), ('mpg','disp'), ('mpg','hp'), ('mpg','wt'), ('mpg','drat'), ('mpg','vs')]

#Qtd de plots será igual ao tamanho do vetor = qtd de pares
n_plots = len(column_pairs)

gig, axes = plt.subplots(nrows = n_plots, ncols = 1, figsize = (5, 2 * n_plots))

#Loop que plota todos os gráficos definidos
for i, pair in enumerate(column_pairs):
    x_col, y_col = pair
    sns.scatterplot(x = x_col, y = y_col, data = base, ax = axes[i])
    
    axes[i].set_title(f'{x_col} vs {y_col}')

#Método para otimizar espaçamento dos gráficos gerados na imagem
plt.tight_layout()
plt.show()

#Método capaz de criar um modelo a partir da definição de variável dependente e independente e uma dada base de dados
#AIC = 155.6 e BIC = 161.4
modelo = sm.ols(formula = 'mpg ~ wt + disp + cyl', data = base)
modelo = modelo.fit()
print(modelo.summary())

#Plotando os resíduos. Meta é obter um formato de sino no gráfico
residuos = modelo.resid

plt.hist(residuos, bins = 20)
plt.xlabel("Residuos")
plt.ylabel("Frequencia")
plt.title("Histograma de residuos")
plt.show()

stats.probplot(residuos, dist = "norm", plot = plt)
plt.title("Q-Q Plot de Residuos")
plt.show()

#h0 - dados estão normalmente distribuídos
#p <= 0.05 rejeito hipótese nula (não estão normalmente distribuídos)
#p > 0.05 não é possível rejeitar a h0
stat, pval = stats.shapiro(residuos)
print(f'Shapiro-Wild estatistica: {stat:.3f}, p-value: {pval:.3f}')
