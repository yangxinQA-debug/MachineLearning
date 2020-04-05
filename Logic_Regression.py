import numpy as np
import cv2
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt


class LogicRegression:
    def generate_data(self):
        iris = datasets.load_iris()
        X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data.astype(np.float32), iris.target.astype(np.float32), test_size=0.1,
                                                                            random_state=42)
        return X_train, X_test, y_train, y_test

    def train_data(self, X_train, X_test, y_train, y_test):
        lr = cv2.ml.LogisticRegression_create()
        lr.setTrainMethod(cv2.ml.LogisticRegression_BATCH)
        lr.setMiniBatchSize(1)
        lr.setIterations(100)
        lr.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
        lr.get_learnt_thetas()
        ret, y_predict = lr.predict(X_train)
        accuracy_score = metrics.accuracy_score(y_train,y_predict)
        print(accuracy_score)
if __name__=="__main__":
    logicRegression = LogicRegression()
    X_train, X_test, y_train, y_test = logicRegression.generate_data()
    logicRegression.train_data(X_train, X_test, y_train, y_test)