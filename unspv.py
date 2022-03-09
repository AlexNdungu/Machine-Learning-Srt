#Exmaple of Unsupervised learning PCA(principal component analysis)

#Choose a class model

from matplotlib.pyplot import axis
from sklearn.decomposition import PCA
import seaborn as sns

#Load Iris Data

iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis = 1)

#Choose Hyperparameters

model = PCA(n_components=2)

#fitting the data

model.fit(X_iris)

#Transform it to 2d data

X_2D = model.transform(X_iris)

print(X_2D)

#Now plot the output

iris['PCA1'] = X_2D[:,0]
iris['PCA2'] = X_2D[:,1]

#The Plot

sns.lmplot('PCA1','PCA2', hue='species',data=iris, fit_reg=False)