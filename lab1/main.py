
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier

from knn import KNearestNeighbor
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

# x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
# y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
#                      np.arange(y_min, y_max, 0.2))

# Z = model.predict(np.c_[xx.ravel(), yy.ravel()] , k = 5 ,num_loops=0 )
# Z = Z.reshape(xx.shape)

# plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)

# # Отобразим точки на графике
# for class_label in set(y):
#     plt.scatter(X[y == class_label][:, 0], X[y == class_label][:, 1], label=f'Class {class_label}')

# # Добавим подписи и легенду
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend(title='Classes')

# # Отобразим график
# plt.show()