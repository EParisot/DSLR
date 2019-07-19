import os
import click
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import read_data, read_model, save_model, get_numerics, classes_list, get_Y, _min, _max

class Predictor(object):

    def __init__(self, data_file, sep, model_file):
        self.model_file = model_file
        self.model = {}
        self.ranges = {}
        self.features = []
        self.classes = []
        # Read data
        self.data, _ = read_data(data_file, sep)
        if len(self.data) == 0:
            print("Error : no valid data found in %s" % data_file)
            exit(0)
        # Read model
        if len(model_file):
            self.model, self.ranges, self.classes = read_model(model_file, self.classes)
            if len(self.model) == 0:
                print("Error : no model found in %s" % model_file)
                exit(0)
        for feat in self.ranges:
            self.features.append(feat)
        
    def clean(self, X):
        clean_X = {}
        for key in X:
            clean_X[key] = []
        for idx, _ in enumerate(X[next(iter(X))]):
            nan = False
            for key in X:
                if X[key][idx] == "NaN":
                    nan = True
                    break
            if nan == False:
                for key in X:
                    clean_X[key].append(X[key][idx])
        return clean_X

    def normalise(self, X):
        norm_X = {}
        for key in X:
            norm_X[key] = []
            for val in X[key]:
                norm_X[key].append((val - self.ranges[key]["min"]) / (self.ranges[key]["max"] - self.ranges[key]["min"]))
        return norm_X

    def select_feat(self):
        X = get_numerics(self.data, False)
        if len(self.features) == 0:
            self.features = []
            for key in X.keys():
                self.features.append(key)
        X_tmp = {}
        for feat in X:
            if feat in self.features:
                X_tmp[feat] = X[feat]
        return X_tmp

    def preprocess(self):
        # select X features:
        X = self.select_feat()
        # clean X Y and normalise X
        clean_X = self.clean(X)
        norm_X = self.normalise(clean_X)
        # append first [ones] to X
        norm_X["ones"] = [1.0] * len(norm_X[next(iter(norm_X))])
        self.features.insert(0, "ones")
        # cast X to np.array
        np_X = np.empty((len(norm_X[next(iter(norm_X))]), len(norm_X)))
        for i, key in enumerate(norm_X):
            for idx, _ in enumerate(norm_X[next(iter(norm_X))]):
                np_X[idx][i] = norm_X[key][idx]
        return np_X

    def predict(self, x):
        return 1.0 / (1 + np.exp(-x))

    def make_prediction(self):
        X = self.preprocess()
        Y = []
        for i, curr_class in enumerate(self.classes):
            # build thetas np array
            thetas = np.zeros((len(self.features), 1))
            for i, theta in enumerate(self.model[curr_class]):
                thetas[i] = self.model[curr_class][theta]
            # predict
            pred = self.predict(np.dot(X, thetas))
            Y.append(pred)
        print(Y)
        pred_by_class = []
        for i in range(len(Y[0])):
            pred = []
            for j in range(len(Y)):
                pred.append(Y[j][i])
            pred_by_class.append(pred)
        # argmax
        for pred in pred_by_class:
            print(pred.index(max(pred)), self.classes[pred.index(max(pred))])


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("-sep", "sep", default=",", help="csv separator")
@click.argument("model_file", default="model.json")
def main(data_file, sep, model_file):
    predictor = Predictor(data_file, sep, model_file)
    predictor.make_prediction()


if __name__ == "__main__":
    main()