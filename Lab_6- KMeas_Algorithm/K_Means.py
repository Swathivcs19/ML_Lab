import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np

data = pd.read_csv("Iris.csv")
data.sample(10)

X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data.Species.astype("category").cat.codes

model = KMeans(n_clusters=3)
model.fit(X)

print('The accuracy score of K-Mean: ', sm.accuracy_score(y, model.labels_))
print('The Confusion matrixof K-Mean: ', sm.confusion_matrix(y, model.labels_))

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 2)
colormap = np.array(['#e74c3c', '#2ecc71', '#9b59b6', '#3498db', '#f1c40f', '#e67e22', '#34495e'])
plt.scatter(X.PetalLengthCm, X.PetalWidthCm, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

K = 1

model = KMeans(n_clusters=K)
model.fit(X)

print('The accuracy score of K-Mean: ', sm.accuracy_score(y, model.labels_))
print('The Confusion matrixof K-Mean: ', sm.confusion_matrix(y, model.labels_))
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 2)
plt.scatter(X.PetalLengthCm, X.PetalWidthCm, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

K = 2

model = KMeans(n_clusters=K)
model.fit(X)

print('The accuracy score of K-Mean: ', sm.accuracy_score(y, model.labels_))
print('The Confusion matrixof K-Mean: ', sm.confusion_matrix(y, model.labels_))

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 2)
plt.scatter(X.PetalLengthCm, X.PetalWidthCm, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

K = 3

model = KMeans(n_clusters=K)
model.fit(X)

print('The accuracy score of K-Mean: ', sm.accuracy_score(y, model.labels_))
print('The Confusion matrixof K-Mean: ', sm.confusion_matrix(y, model.labels_))

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 2)
plt.scatter(X.PetalLengthCm, X.PetalWidthCm, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

K = 4

model = KMeans(n_clusters=K)
model.fit(X)

print('The accuracy score of K-Mean: ', sm.accuracy_score(y, model.labels_))
print('The Confusion matrixof K-Mean: ', sm.confusion_matrix(y, model.labels_))

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 2)
plt.scatter(X.PetalLengthCm, X.PetalWidthCm, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

K = 5

model = KMeans(n_clusters=K)
model.fit(X)

print('The accuracy score of K-Mean: ', sm.accuracy_score(y, model.labels_))
print('The Confusion matrixof K-Mean: ', sm.confusion_matrix(y, model.labels_))

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 2)
plt.scatter(X.PetalLengthCm, X.PetalWidthCm, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

K = 6

model = KMeans(n_clusters=K)
model.fit(X)

print('The accuracy score of K-Mean: ', sm.accuracy_score(y, model.labels_))
print('The Confusion matrixof K-Mean: ', sm.confusion_matrix(y, model.labels_))

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 2)
plt.scatter(X.PetalLengthCm, X.PetalWidthCm, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

K = 7

model = KMeans(n_clusters=K)
model.fit(X)

print('The accuracy score of K-Mean: ', sm.accuracy_score(y, model.labels_))
print('The Confusion matrixof K-Mean: ', sm.confusion_matrix(y, model.labels_))

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 2)
plt.scatter(X.PetalLengthCm, X.PetalWidthCm, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

