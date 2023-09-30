
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier

wine = load_wine()
X = wine.data[:, :2]  # Возьмем только первые два признака для простоты
y = wine.target

X_train = X[:150]
y_train = y[:150]
X_test = X[151:]
y_test =y[151:]

model = KNearestNeighbor()
model.fit(X_train,y_train)
knn_sklearn = KNeighborsClassifier(n_neighbors=5)
knn_sklearn.fit(X_train , y_train)

knn_sklearn_preds = knn_sklearn.predict(X_test )
model_preds = model.predict(X_test, k=5)
print(knn_sklearn_preds)
print(model_preds)

print(np.array_equal(knn_sklearn_preds, model_preds))

data = X
target = y

