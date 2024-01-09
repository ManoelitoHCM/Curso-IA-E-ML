from numpy import *

class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.__correlation_coefficient = self.__correlacao()
        self.__inclination = self.__inclinacao()
        self.__intercept = self.__interceptacao()
    
    #Calcula a correlaçao entre as variaveis
    def __correlacao(self):
        covariacao = cov(self.x, self.y, bias = True)[0][1]
        
        variancia_x = var(self.x)
        variancia_y = var(self.y)
        
        return covariacao / sqrt(variancia_x * variancia_y)
        
    #Calcula a inclinaçao da reta que define a regressao 
    def __inclinacao(self):
        stdx = std(self.x)
        stdy = std(self.y)
        
        return self.__correlation_coefficient * (stdy / stdx)
    
    #Calcula a interceptacao na reta (valor de y quando x = 0) 
    def __interceptacao(self):
        mediax = mean(self.x)
        mediay = mean(self.y)
        
        return mediay - mediax * self.__inclination
    
    #Metodo para estimar determinado valor com base na regressao feita
    def previsao(self, valor):
        return self.__intercept + (self.__inclination * valor)
    
x = array([1, 2, 3, 4, 5])
y = array([2, 4, 6, 8, 10])

lr = LinearRegression(x,y)

previsao = lr.previsao(9)

print(previsao)