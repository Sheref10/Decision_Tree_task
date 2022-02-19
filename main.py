import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import pandas_profiling as pp
import matplotlib
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from dtreeviz.trees import dtreeviz
matplotlib.use('tkagg')
#Load the dataset
iris=datasets.load_iris()
iris_data=pd.DataFrame(iris.data,columns=iris.feature_names)
target=iris.target
print(target)
#Data exploration
print(iris_data.info)
print(iris_data.describe())
report=pp.ProfileReport(iris_data)
#report.to_file('output.html')
#
# #Data Cleaning
print(iris_data.duplicated().sum())
#print(iris_data.drop_duplicates(inplace=True))
print(iris_data.duplicated().sum())
print(iris_data.isnull().sum())
iris_data.corr().style.background_gradient(cmap='green')
iris_data.boxplot()
plt.show()
# Modeling the decision tree algorithm
Decision_Tree=DecisionTreeClassifier()
Decision_Tree.fit(iris_data,target)
print('Decision Tree Classifer Created')

#Decision Tree visualization
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(Decision_Tree,
                   feature_names=iris.feature_names,
                   class_names=iris.target_names,
                   filled=True)
fig.savefig("decistion_tree.png")
plt.show()
