import numpy as np
from sklearn import random_projection
range = np.random.RandomState(0)

X = range.rand(10,2000)

X = np.array(X,dtype='float32')

print(X.dtype)

#Now we transform the data

transformer = random_projection.GaussianRandomProjection()

X_new = transformer.fit_transform(X)

print(X_new.dtype)