import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn import model_selection as ms
import numpy as np
from sklearn import tree


class DecisionTree:
    def generate_data(self):
        data = [
            {'age': 33, 'sex': 'F', 'BP': 'high', 'cholesterol': 'high', 'Na': 0.66, 'K': 0.06, 'drug': 'A'},
            {'age': 77, 'sex': 'F', 'BP': 'high', 'cholesterol': 'normal', 'Na': 0.19, 'K': 0.03, 'drug': 'D'},
            {'age': 88, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.80, 'K': 0.05, 'drug': 'B'},
            {'age': 39, 'sex': 'F', 'BP': 'low', 'cholesterol': 'normal', 'Na': 0.19, 'K': 0.02, 'drug': 'C'},
            {'age': 43, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'high', 'Na': 0.36, 'K': 0.03, 'drug': 'D'},
            {'age': 82, 'sex': 'F', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.09, 'K': 0.09, 'drug': 'C'},
            {'age': 40, 'sex': 'M', 'BP': 'high', 'cholesterol': 'normal', 'Na': 0.89, 'K': 0.02, 'drug': 'A'},
            {'age': 88, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.80, 'K': 0.05, 'drug': 'B'},
            {'age': 29, 'sex': 'F', 'BP': 'high', 'cholesterol': 'normal', 'Na': 0.35, 'K': 0.04, 'drug': 'D'},
            {'age': 53, 'sex': 'F', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.54, 'K': 0.06, 'drug': 'C'},
            {'age': 36, 'sex': 'F', 'BP': 'high', 'cholesterol': 'high', 'Na': 0.53, 'K': 0.05, 'drug': 'A'},
            {'age': 63, 'sex': 'M', 'BP': 'low', 'cholesterol': 'high', 'Na': 0.86, 'K': 0.09, 'drug': 'B'},
            {'age': 60, 'sex': 'M', 'BP': 'low', 'cholesterol': 'normal', 'Na': 0.66, 'K': 0.04, 'drug': 'C'},
            {'age': 55, 'sex': 'M', 'BP': 'high', 'cholesterol': 'high', 'Na': 0.82, 'K': 0.04, 'drug': 'B'},
            {'age': 35, 'sex': 'F', 'BP': 'normal', 'cholesterol': 'high', 'Na': 0.27, 'K': 0.03, 'drug': 'D'},
            {'age': 23, 'sex': 'F', 'BP': 'high', 'cholesterol': 'high', 'Na': 0.55, 'K': 0.08, 'drug': 'A'},
            {'age': 49, 'sex': 'F', 'BP': 'low', 'cholesterol': 'normal', 'Na': 0.27, 'K': 0.05, 'drug': 'C'},
            {'age': 27, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.77, 'K': 0.02, 'drug': 'B'},
            {'age': 51, 'sex': 'F', 'BP': 'low', 'cholesterol': 'high', 'Na': 0.20, 'K': 0.02, 'drug': 'D'},
            {'age': 38, 'sex': 'M', 'BP': 'high', 'cholesterol': 'normal', 'Na': 0.78, 'K': 0.05, 'drug': 'A'}
        ]
        target = [d['drug'] for d in data]
        # ABCD 字母变成数字
        target = [ord(t) - 65 for t in target]
        [d.pop('drug') for d in data]
        # 预处理
        vec = DictVectorizer(sparse=False)
        print(data)
        data_pre = vec.fit_transform(data)
        data_pre = np.array(data_pre, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        X_train, X_test, y_train, y_test = ms.train_test_split(data_pre, target, test_size=5, random_state=42)
        return X_train, X_test, y_train, y_test,vec

    def train_data(self, X_train, X_test, y_train, y_test):
        # dtc = tree.DecisionTreeClassifier()
        # dtc.fit(X_train, y_train)
        # print(dtc.score(X_train, y_train))
        # print(dtc.score(X_test, y_test))

        dtc0 = tree.DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=6)
        dtc0.fit(X_train, y_train)
        print(dtc0.score(X_train, y_train))
        print(dtc0.score(X_test, y_test))

        return dtc0

    def plot_data(self, dtc, vec):
        with open("tree.dot", "w") as f:
            f = tree.export_graphviz(dtc,out_file=f,feature_names=vec.get_feature_names(), class_names=['A', 'B', 'C', 'D'])


if __name__=="__main__":
    decision_tree = DecisionTree()
    X_train, X_test, y_train, y_test,vec = decision_tree.generate_data()
    dtc =  decision_tree.train_data( X_train, X_test, y_train, y_test)
    decision_tree.plot_data(dtc,vec)
