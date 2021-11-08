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
pca1 = len(values_raw[values_raw >3])

print(pca1)
print(pca1-1)


pca = PCA(pca1-1).fit(standardized_trainx)
pca_trainx = pca.transform(standardized_trainx)
pca_testx = pca.transform(transformed_testx)

pca = PCA(n_components = 0.8)
pca.fit(standardized_trainx)

pca.n_components_

new_trainx = pca.transform(standardized_trainx)
new_testx = pca.transform(transformed_testx)


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(new_trainx, nptrain_y)

pred = classifier.predict(new_testx)


from sklearn.model_selection import  cross_val_score

ac_score = cross_val_score(classifier, new_trainx, nptrain_y, cv =3, scoring = "accuracy")

# 2Dimension 
print("-----Random Forest 2D-data-----")
print(ac_score.max())