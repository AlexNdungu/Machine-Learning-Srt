#SUPERVISED LEARNING EXAMPLE
#Regression Example

import matplotlib.pyplot as plt
import numpy as np

#Generate random numbers using RandomState

rng = np.random.RandomState(35)

x = 10*rng.rand(40)

y = 2*x-1+rng.randn(40)

#Plot This output

plt.scatter(x,y)

#Choose A linear Regression Class

from sklearn.linear_model import LinearRegression

#choose HydraParameters

model = LinearRegression(fit_intercept=True)

#Arrange Data

X = x[:, np.newaxis]

theS = X.shape

print(theS)

#Now we fit our data

model.fit(X,y)

theCof = model.coef_

print(theCof)


#Now we Test The new model using predict() method

xfit = np.linspace(-1,11)

Xfit = xfit[:,np.newaxis]

yfit = model.predict(Xfit)

#Outputing the graph

plt.scatter(x,y)

plt.plot(xfit,yfit)


