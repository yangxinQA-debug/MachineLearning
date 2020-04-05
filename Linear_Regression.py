import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import linear_model
import matplotlib.pyplot as plt


class Linear_Regression:
    def generate_data(self):
        boston = datasets.load_boston()
        X_train, X_test, y_train, y_test = model_selection.train_test_split(boston.data, boston.target, test_size=0.1,
                                                                            random_state=42)
        return X_train, X_test, y_train, y_test

    def train_data(self, X_train, X_test, y_train, y_test):
        linreg = linear_model.LinearRegression()
        linreg.fit(X_train, y_train)
        train_mean_squared_error = metrics.mean_squared_error(y_train, linreg.predict(X_train))
        train_q2 = linreg.score(X_train, y_train)
        print(train_mean_squared_error)
        print(train_q2)
        y_pred = linreg.predict(X_test)
        test_mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
        return y_pred

    def plot_data(self, X_train, X_test, y_train, y_test, y_pred):
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, linewidth=3, label="ground_truth")
        plt.plot(y_pred, linewidth=3, label="predicted")
        #plt.legend(loc="test")
        plt.xlabel("test data points")
        plt.ylabel("target value")


if __name__ == "__main__":
    linreg = Linear_Regression()
    X_train, X_test, y_train, y_test = linreg.generate_data()
    y_pred = linreg.train_data(X_train, X_test, y_train, y_test)
    linreg.plot_data(X_train, X_test, y_train, y_test, y_pred)
