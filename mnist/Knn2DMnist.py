from operator import imatmul
from main import load_data
from pca import PCA
import numpy as np
import pandas as pd
from collections import Counter


if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')

    train_x, train_y = train_set
    val_x, val_y = val_set
    test_x, test_y = test_set
    
print("---Original Data shape--")
print(train_x.shape)
print(train_y.shape) # shape


rint = np.random.randint(train_x.shape[0], size = 10000) # 10000
nptrain_x = train_x[rint, :]
nptrain_y = train_y[rint]

print("---Random train 10000 sample Data shape--")
print(nptrain_x.shape)
print(nptrain_y.shape)


rint = np.random.randint(test_x.shape[0], size = 20000) # 20000
nptest_x = test_x[rint, :]
nptest_y = test_y[rint]

print("---Random test 20000 sample Data shape--")
print(nptest_x.shape)
print(nptest_y.shape)


# Data-preprocessing: Standardizing the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler_fit = scaler.fit(nptrain_x)
standardized_trainx = scaler_fit.transform(nptrain_x)
transformed_testx = scaler.transform(nptest_x)

# np.cov

cov_mat = np.cov(nptrain_x.T)
values_raw, components_raw = np.linalg.eig(cov_mat) # eigenvalue 
pca1 = len(values_raw[values_raw > 3])
print(pca1-1)

pca = PCA(pca1-1).fit(standardized_trainx)
pca_trainx = pca.transform(standardized_trainx)
pca_testx = pca.transform(transformed_testx)

pca = PCA(n_components=0.8)

pca.fit(standardized_trainx)

pca.n_components_

pca = PCA(n_components= 120)
pca.fit(standardized_trainx)


new_trainx = pca.transform(standardized_trainx)
new_testx = pca.transform(transformed_testx)



# Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(nptrain_x)
X_projected = pca.transform(nptrain_x)

print("Shape of X:", nptrain_x.shape)
print("Shape of transformed X:", X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]


'''
plt.figure(figsize= (8, 6))
plt.scatter(
    x1, x2, c = labels, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 10)
)
plt.title("PCA scatter")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

'''



def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

if __name__ == "__main__":
    
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy


k = 10
clf = KNN(k=k)
clf.fit(new_trainx, nptrain_y)
predictions = clf.predict(new_testx)
print("KNN classification accuracy(2D-data, K = 10)", accuracy(nptest_y, predictions))

k = 5
clf = KNN(k=k)
clf.fit(new_trainx, nptrain_y)
predictions = clf.predict(new_testx)
print("KNN classification accuracy(2D-data, K = 5)", accuracy(nptest_y, predictions))

k = 1
clf = KNN(k=k)
clf.fit(new_trainx, nptrain_y)
predictions = clf.predict(new_testx)
print("KNN classification accuracy(2D-data, K = 1)", accuracy(nptest_y, predictions))


