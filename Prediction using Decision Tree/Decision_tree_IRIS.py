import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import tree

#load dataset
iris_data = load_iris()
iris = pd.DataFrame(iris_data.data)

#print features
print ("Features Name : " , iris_data.feature_names)

#shape
print ("Dataset Shape : ", iris.shape)

print ("Dataset : ", iris.head())

#printing sample and targets
X= iris.values[:, 0:4]
Y = iris_data.target
print (X)
print (Y)
 
#splitting the data set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 100)

#decision tree classifier
clf = DecisionTreeClassifier(random_state = 100)

#fitting Train data
clf.fit(X_train, y_train)

#prediction on random
X = [[6.4,1.7 ,6.5,2.3]]
Y_pred=clf.predict(X)
print (Y_pred)

#prediction on X_test
Y_pred=clf.predict(X_test)
print(Y_pred)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = np.array(confusion_matrix(y_test, Y_pred))
print(cm)

#tree plot
tree.plot_tree(clf)

#decision making
text_representation = tree.export_text(clf)
print(text_representation)















