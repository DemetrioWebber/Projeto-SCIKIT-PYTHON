# Iris dataset
# importação dos dados
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_iris
iris = load_iris()

#Observações
x = iris.data
#Target
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)

valores_performance = {}

for k in range(1, 26):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    species = knn.predict([[5.9, 3,  5.1, 1.8]])[0]
    previsoes = knn.predict(x_test)
    acertos = metrics.accuracy_score(y_test, previsoes)
    valores_performance[k] = round(acertos, 3)

import matplotlib.pyplot as plt

plt.plot(list(valores_performance.keys()), list(valores_performance.values()))
plt.xlabel("Valores de K")
plt.ylabel("Performance")
plt.show()