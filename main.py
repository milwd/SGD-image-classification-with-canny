
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from funks import getData, trainPrep, show
import time


data_path = '../self-images'
width, height = 100, 100
mustInclude = {'hoodie', 'monitor', 'sandwich'}
split = 0.3
max_iter = 1000
tol = 1e-3
loss = 'hinge'
randomState = 21

data = getData(src=data_path, wid=width, hei=height, include=mustInclude)

X = np.array(data['data'])
y = np.array(data['label'])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=True)
print(f'\n data split: {100 * round(1-split, 2)}% train -- {100 * round(split, 2)}% test !')
X_train = np.copy(x_train)
X_test = np.copy(x_test)
X_train_prepared = trainPrep(X_train)
X_test_prepared = trainPrep(X_test)

sgd = SGDClassifier(max_iter=max_iter, tol=tol, loss=loss, random_state=21)
print(f'\n Classifier initialized: epochs= {max_iter}, stopping criterion= {tol}, loss= {loss}, random state = {randomState}')
initTime = time.time()
sgd.fit(X_train_prepared, y_train)
endTime = time.time()
print(f'\n Training completed, time taken: {round(endTime-initTime, 2)} seconds !')
score = sgd.score(X_test_prepared, y_test)
print(f'\n Total score: {round(score, 2)}')
y_pred = sgd.predict(X_test_prepared)
show(x_test, y_test, y_pred, score)
