

from operator import imatmul
from main import load_data
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import*
import numpy as np

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


# using randomforest classifier

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(nptrain_x, nptrain_y)

pred = classifier.predict(nptrain_x)


from sklearn.model_selection import  cross_val_score

ac_score = cross_val_score(classifier, nptrain_x, nptrain_y, cv =3, scoring = "accuracy")

# Raw dimension
print("-----Random Forest Rawdata-----")
print(ac_score.max())

'''
# for Hyperparameter tuning 
from sklearn.model_selection import GridSearchCV

params = { 'n_estimators' : [10, 100],
           'max_depth' : [6, 8, 10, 12],
           'min_samples_leaf' : [8, 12, 18],
           'min_samples_split' : [8, 16, 20]
            }

rf_clf = RandomForestClassifier(random_state = 0, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(nptrain_x, nptrain_y)

print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))

#위의 결과로 나온 최적 하이퍼 파라미터로 다시 모델을 학습하여 테스트 세트 데이터에서 예측 성능을 측정
rf_clf1 = RandomForestClassifier(n_estimators = 100, 
                                max_depth = 12,
                                min_samples_leaf = 8,
                                min_samples_split = 8,
                                random_state = 0,
                                n_jobs = -1)
rf_clf1.fit(nptrain_x, nptrain_y)
pred = rf_clf1.predict(nptest_x)
print('예측 정확도: {:.4f}'.format(accuracy_score(nptest_y,pred)))

'''