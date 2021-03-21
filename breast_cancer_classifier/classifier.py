from sklearn.datasets import load_breast_cancer
import pandas as pd


df  = load_breast_cancer()
print(df.data)
print(df.feature_names)
print(df.target)
print(df.target_names)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.data, df.target,test_size = 0.2, random_state = 12)
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))
neigh = []
from sklearn.neighbors import KNeighborsClassifier
for k in range(1, 100):
  knn_model = KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train)
  tahmin = knn_model.predict(x_test)
  neigh.append(knn_model.score(x_test, y_test))

import matplotlib.pyplot as plt

plt.plot(range(1,100), neigh, label="normal data")


from sklearn.preprocessing import StandardScaler
scaled_x_train = StandardScaler().fit_transform(x_train)
scaled_x_test = StandardScaler().fit_transform(x_test)

neigh2 = []
for k in range(1, 100):
  knn_model = KNeighborsClassifier(n_neighbors = k).fit(scaled_x_train, y_train)
  neigh2.append(knn_model.score(scaled_x_test, y_test))

plt.plot(range(1,100), neigh2, label="scaled data")
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

