import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os
from sklearn import model_selection as ms
from sklearn import metrics
from matplotlib import patches

class Detect_Pedestrain:
    def prepare_data(self):
        data_dir = "data"
        dataset = "pedestrians128x64"
        datafile = "%s/%s.tar.gz" % (data_dir, dataset)
        extractdir = "%s/%s" % (data_dir, dataset)
        self._extract_tar(datafile, data_dir)
        self._draw_file_image(extractdir)

        win_size = (48, 96)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        num_bins = 9
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
        random.seed(42)
        X_pos = []
        for i in random.sample(range(900), 400):
            filename = "%s/per%05d.ppm" % (extractdir, i)
            img = cv2.imread(filename)
            if img is None:
                print('Could not find image %s' % filename)
                continue
            X_pos.append(hog.compute(img, (64, 64)))

        X_pos = np.array(X_pos, dtype=np.float32)
        y_pos = np.ones(X_pos.shape[0], dtype=np.int32)
        print(X_pos.shape, y_pos.shape)

        negset = "pedestrians_neg"
        negfile = "%s/%s.tar.gz" % (data_dir, negset)
        negdir = "%s/%s" % (data_dir, negset)
        self._extract_tar(negfile, data_dir)

        hroi = 128
        wroi = 64
        X_neg = []
        for negfile in os.listdir(negdir):
            filename = "%s/%s" % (negdir, negfile)
            img = cv2.imread(filename)
            img = cv2.resize(img, (512, 512))
            for j in range(5):
                rand_y = random.randint(0, img.shape[0] - hroi)
                rand_x = random.randint(0, img.shape[1] - wroi)
                roi = img[rand_y:rand_y + hroi, rand_x:rand_x + wroi, :]
                X_neg.append(hog.compute(roi, (64, 64)))

            X_neg = np.array(X_neg, dtype=np.float32)
            y_neg = -np.ones(X_neg.shape[0], dtype=np.int32)

            X = np.concatenate((X_pos, X_neg))
            y = np.concatenate((y_pos, y_neg))

            X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test, hog

    def _extract_tar(self, datafile, extractdir):
        try:
            import tarfile
        except ImportError:
            raise ImportError("You do not have tarfile installed. "
                              "Try unzipping the file outside of Python.")
        tar = tarfile.open(datafile)
        tar.extractall(path=extractdir)
        tar.close()
        print("%s successfully extracted to %s" % (datafile, extractdir))

    def _draw_file_image(self, extractdir):
        plt.figure(figsize=(10, 6))
        for i in range(5):
            filename = "%s/per0010%d.ppm" % (extractdir, i)
            img = cv2.imread(filename)
            plt.subplot(1, 5, i + 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis("off")

    def train_svm(self, X_train, y_train, X_test, y_test):
        svm = cv2.ml.SVM_create()
        # svm.train(X_train, ytrain)
        score_train = []
        score_test = []
        for j in range(3):
            svm.train(X_train,cv2.ml.ROW_SAMPLE,  y_train)
            print(X_train.shape,y_train.shape)
            _, y_predict = svm.predict(X_train)
            score_train.append(metrics.accuracy_score(y_train, y_predict))
            _, y_predict = svm.predict(X_test)
            score_test.append(metrics.accuracy_score(y_test, y_predict))
            _, y_predict = svm.predict(X_test)
            false_pos = np.logical_and((y_test.ravel() == -1), (y_predict.ravel() == 1))
            if not np.any(false_pos):
                print("done")
            X_train = np.concatenate((X_train, X_test[false_pos, :]), axis=0)
            y_train = np.concatenate((y_train, y_test[false_pos]), axis=0)
        return svm

    def predict_data(self, svm, X_train, X_test, y_train, y_test):
        _, y_predict = svm.predict(X_train)
        train_score = metrics.accuracy_score(y_train, y_predict)
        print("train score: " + str(train_score))
        _, y_predict = svm.predict(X_test)
        test_score = metrics.accuracy_score(y_test, y_predict)
        print("test score: " + str(test_score))

    def detect_pedestrain(self, hog, svm):
        win_size = (48, 96)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        num_bins = 9
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
        rho, _, _ = svm.getDecisionFunction(0)
        sv = svm.getSupportVectors()
        hog.setSVMDetector(np.append(sv[0, :].ravel(), rho))
        img_test = cv2.imread('data/pedestrian_test.jpg')
        stride = 16
        hroi = 128
        wroi = 64
        found = []
        for ystart in np.arange(0, img_test.shape[0], stride):
            for xstart in np.arange(0, img_test.shape[1], stride):
                if ystart + hroi > img_test.shape[0]:
                    continue
                if xstart + wroi > img_test.shape[1]:
                    continue
                roi = img_test[ystart:ystart + hroi, xstart:xstart + hroi,:]
                feat = np.array([hog.compute(roi, (64, 64))])
                _, ypred = svm.predict(feat)
                if np.allclose(ypred, 1):
                    found.append((ystart, xstart, hroi, wroi))
        return found

    def draw_result(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
        for f in found:
            ax.add_patch(patches.Rectangle((f[0], f[1]), f[2], f[3], color='y', linewidth=3, fill=False))
        plt.savefig('detected.png')


if __name__ == "__main__":
    img_test = cv2.imread('data/chapter6/pedestrian_test.jpg')
    dp = Detect_Pedestrain()
    X_train, X_test, y_train, y_test, hog = dp.prepare_data()
    svm = dp.train_svm(X_train, y_train, X_test, y_test)
    dp.predict_data(svm, X_train, X_test, y_train, y_test)
    found = dp.detect_pedestrain(hog, svm)
