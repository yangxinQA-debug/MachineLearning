import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_extraction import DictVectorizer
from sklearn import model_selection as ms
import numpy as np
from sklearn import tree

class DecisionTreeForCancer:
    def generateData(self):
        data = datasets.load_breast_cancer()
        X_train,X_test,y_train,y_test = ms.train_test_split(data.data,data.target,test_size=0.2,random_state=42)
        return X_train,X_test,y_train,y_test,data

    def trainData(self,X_train,X_test,y_train,y_test):
        dtc = tree.DecisionTreeClassifier(random_state=42)
        dtc.fit(X_train,y_train)
        print(dtc.score(X_train, y_train))
        print(dtc.score(X_test, y_test))
        return dtc

    def plot_data(self, dtc,data):
        with open("tree2.dot", 'w') as f:
            f = tree.export_graphviz(dtc, out_file=f,
                                     feature_names=data.feature_names,
                                     class_names=data.target_names)
if __name__=="__main__":
    decision_tree = DecisionTreeForCancer()
    X_train, X_test, y_train, y_test,data = decision_tree.generateData()
    dtc =  decision_tree.trainData( X_train, X_test, y_train, y_test)
    decision_tree.plot_data(dtc,data)

