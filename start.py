#Sklearn Loading Data

#The iris dataset(classification)

from scipy import rand
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print('Feature names:', feature_names)
print('target names:', target_names)
print('\nFirst 10 rows of X:\n',X[:10])

#Now we split this data to training dataset and testing dataset

#For spliting we use train_test_split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.3, random_state=1)

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)


#Trainig Your Model
#KNN algrithim

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

classifier_knn = KNeighborsClassifier(n_neighbors= 3)
classifier_knn.fit(X_train,y_train)

#Now we predict the data
y_pred = classifier_knn.predict(X_test)

#Finding accuracy by comparing actual response valie (y_test) with predictedresponse value (y_pred)
print('Accuracy:', metrics.accuracy_score(y_test,y_pred))

#Let us giv eit a sample to predict from
sample = [[5,5,3,2],[2,4,3,5]]
preds = classifier_knn.predict(sample)

pred_species = [iris.target_names[p] for p in preds]

print('predicts:', pred_species)

#The trained mode need to be persistant for the future
from sklearn.externals import joblib

joblib.dump(classifier_knn, 'iris_classifier_knn.joblib')

#the object needs to reloaded in the future
joblib.load('iris_classifier_knn.joblib')

