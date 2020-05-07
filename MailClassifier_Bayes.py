import os
import pandas as pd
from sklearn import feature_extraction
from sklearn import model_selection as ms
import cv2
from sklearn import naive_bayes


class MailClassifier_Bayes(object):
    def prepare_data(self):
        HAM = 0
        SPAM = 1
        datadir = 'data/chapter7'
        sources = [
            ('beck-s.tar.gz', HAM),
            ('farmer-d.tar.gz', HAM),
            ('kaminski-v.tar.gz', HAM),
            ('kitchen-l.tar.gz', HAM),
            ('lokay-m.tar.gz', HAM),
            ('williams-w3.tar.gz', HAM),
            ('BG.tar.gz', SPAM),
            ('GP.tar.gz', SPAM),
            ('SH.tar.gz', SPAM)
        ]

        for source, _ in sources:
            datafile = "%s/%s" % (datadir, source)
            self.extract_tar(datafile, datadir)

        data = pd.DataFrame({'text': [], 'class': []})
        for source, classification in sources:
            extractdir = "%s/%s" % (datadir, source[:-7])
            data = data.append(self.build_data_frame(extractdir, classification))
        counts = feature_extraction.text.CountVectorizer()
        X = counts.fit_transform(data['text'].values)
        y = data['class'].values
        X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, X_test, y_train, y_test):
        model_naive = naive_bayes.MultinomialNB()
        model_naive.fit(X_train, y_train)
        print(model_naive.score(X_train, y_train))
        print(model_naive.score(X_test, y_test))

    def extract_tar(self, datafile, extractdir):
        try:
            import tarfile
        except ImportError:
            raise ImportError("You do not have tarfile installed")
        tar = tarfile.open(datafile)
        tar.extractall(path=extractdir)
        tar.close()
        print("%s successfully extracted to %s" % (datafile, extractdir))

    def read_signle_file(self, filename):
        past_header, lines = False, []
        if os.path.isfile(filename):
            f = open(filename, encoding="latin-1")
            for line in f:
                if past_header:
                    lines.append(line)
                elif line == "\n":
                    past_header = True
            f.close()
        content = "\n".join(lines)
        return filename, content

    def read_files(self, path):
        for root, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                yield self.read_signle_file(filepath)

    def build_data_frame(self, extractdr, classification):
        rows = []
        index = []
        for file_name, text in self.read_files(extractdr):
            rows.append({'text': text, 'class': classification})
            index.append(file_name)
        data_frame = pd.DataFrame(rows, index=index)
        return data_frame


if __name__ == "__main__":
    byes = MailClassifier_Bayes()
    X_train, X_test, y_train, y_test = byes.prepare_data()
    byes.train_model( X_train, X_test, y_train, y_test)