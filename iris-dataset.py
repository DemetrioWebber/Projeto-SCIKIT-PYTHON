# Iris dataset
# importação dos dados

from sklearn.datasets import load_iris
iris = load_iris()

#Observações
x = iris.data
print(x)

#Target
y = iris.target
print(y)


#Shape das observações (Se não fossem arrays numpy, poderia usar o Len, mas como são tem que usar o shape)
print(iris.data.shape)

#Shape do target
print(iris.target.shape)

# importação do KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# Treinar a máquina
knn.fit(x,y)

#Fazer previsões
species = knn.predict([[5.9, 3,  5.1, 1.8]])[0]
print(iris.target_names[species])

#Separar os dados em dois grupos (Para ter certeza de que quando o pc for fazer a previsão, ele faça com  dados que não viu antes)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)

#Avaliação da performance do modelo
knn.fit(x_train, y_train)
previsoes = knn.predict(x_test)

from sklearn import metrics
acertos = metrics.accuracy_score(y_test, previsoes)
print(acertos)