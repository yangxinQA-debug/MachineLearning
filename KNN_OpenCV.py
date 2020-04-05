import numpy as np
import cv2
import matplotlib.pyplot as plt


class Knn_Opencv:

    def generate_data(self, num_samples, num_features=2):
        np.random.seed(42)
        data_size = (num_samples, num_features)
        train_data = np.random.randint(0, 100, size=data_size)
        label_size = (num_samples, 1)
        labels = np.random.randint(0, 2, size=label_size)
        return train_data.astype(np.float32), labels

    def plot_data(self, all_blue, all_red):
        plt.figure(figsize=(10, 6))
        plt.scatter(all_blue[:, 0], all_blue[:, 1], c="b", marker="s", s=180)
        plt.scatter(all_red[:, 0], all_red[:, 1], c="r", marker="^", s=180)
        plt.xlabel("x coordinate(feature1)")
        plt.ylabel("y coordinate(feature2)")

    def train_data(self,train_data, labels):
        knn = cv2.ml.KNearest_create()
        knn.train(train_data, cv2.ml.ROW_SAMPLE, labels)
        new_comer = self.generate_data(1)
        plt.plot(new_comer[0][0][0], new_comer[0][0][1], "go", markersize=14)
        ret, results, neighbor, dist = knn.findNearest(new_comer[0], 1)
        print("predict label:", results)
        print("neighbour label:", neighbor)
        print("distance to neighbour", dist)


if __name__ == "__main__":
    knn_opencv = Knn_Opencv()
    train_data, labels = knn_opencv.generate_data(11)
    print(train_data)
    blue = train_data[labels.ravel() == 0]
    red = train_data[labels.ravel() == 1]
    knn_opencv.plot_data(blue, red)
    knn_opencv.train_data(train_data, labels)
