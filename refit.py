import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC

X,y = load_iris(return_X_y=True)

clf = SVC()

clf.set_params(kernel = 'linear').fit(X,y)

clf.predict(X[:5])

print(clf.predict(X[:5]))