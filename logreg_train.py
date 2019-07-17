import os
import click
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import read_data, read_model, save_model, get_numerics, get_classes, get_Y, _min, _max

class Trainer(object):

    def __init__(self, data_file, features, sep, model_file, plot, epochs, lr):
        self.model_file = model_file
        self.features = list(features)
        self.plot = plot
        self.epochs = epochs
        self.model = {}
        self.ranges = {}
        self.lr = lr
        self.acc = []
        self.loss = []
        # Read data
        self.data = read_data(data_file, sep)
        if len(self.data) == 0:
            print("Error : no valid data found in %s" % data_file)
            exit(0)
        self.classes_column = "Hogwarts House"
        self.classes = get_classes(self.data, self.classes_column)
        # Read model
        if len(model_file):
            self.model, _ = read_model(model_file, self.classes)
        
    def clean(self, X, Y):
        clean_X = {}
        clean_Y = []
        for key in X:
            clean_X[key] = []
        for idx, _ in enumerate(X[next(iter(X))]):
            nan = False
            for key in X:
                if X[key][idx] == "NaN":
                    nan = True
                    break
            if nan == False:
                clean_Y.append(Y[idx])
                for key in X:
                    clean_X[key].append(X[key][idx])
        return clean_X, clean_Y

    def normalise(self, X):
        norm_X = {}
        for key in X:
            norm_X[key] = []
            self.ranges[key] = {}
            k_min = _min(X[key])
            k_max = _max(X[key])
            self.ranges[key]["min"] = k_min
            self.ranges[key]["max"] = k_max
            for val in X[key]:
                norm_X[key].append((val - k_min) / (k_max - k_min))
        return norm_X

    def train(self):
        # buid Y
        Y = get_Y(self.data, self.classes_column)
        # select X features:
        X = get_numerics(self.data, False)
        if len(self.features) == 0:
            self.features = []
            for key in X.keys():
                self.features.append(key)
        X_tmp = {}
        for feat in X:
            if feat in self.features:
                X_tmp[feat] = X[feat]
        X = X_tmp
        # clean X Y and normalise X
        clean_X, clean_Y = self.clean(X, Y)
        norm_X = self.normalise(clean_X)
        # append first [ones] to X
        norm_X["ones"] = [1.0] * len(clean_Y)
        self.features.insert(0, "ones")
        # cast X to np.array
        np_X = np.empty((len(clean_Y), len(norm_X)))
        for i, key in enumerate(self.features):
            for idx, _ in enumerate(clean_Y):
                np_X[idx][i] = norm_X[key][idx]
        # one vs all
        for curr_class in self.classes:
            # build tmp Y (one v all)
            tmp_Y = np.zeros((len(clean_Y), 1))
            for i, val in enumerate(clean_Y):
                if val == curr_class:
                    tmp_Y[i] = 1
            # build thetas np array
            thetas = np.zeros((len(self.features), 1))
            for i, theta in enumerate(self.model[curr_class]):
                thetas[i] = self.model[curr_class][theta]
            # train
            self.train_loop(np_X, tmp_Y, thetas, curr_class)
            # save model
            save_model(self.model, self.ranges, self.model_file)
        # plot result
        if self.plot:
            plt.figure("Train history")
            plt.plot(self.acc, label="acc")
            plt.plot(self.loss, label="loss")
            plt.legend()
            plt.show(block=True)
    
    def train_loop(self, X, Y, thetas, curr_class):
        # loop on epochs / batches / data_points
        for epoch in range(self.epochs):
            print("Training... Epoch : %d" % (epoch + 1))
            loss, acc = self.train_epoch(X, Y, thetas, curr_class)
            self.acc.append(acc)
            self.loss.append(loss)
            # print
            print("loss : %f ; acc : %f" % (round(loss, 2), round(acc, 2)))
    
    def train_epoch(self, X, Y, thetas, curr_class):
        # train
        pred = self.predict(np.dot(X, thetas))
        loss = self.cost_func(Y, pred)
        thetas -= (self.lr * np.dot(X.T, (pred - Y)))
        # save thetas
        for i, theta in enumerate(thetas):
            self.model[curr_class]["theta_" + str(i)] = theta[0]
        # adjust lr
        
        # metrics
        acc = 0
        return loss, acc

    def predict(self, x):
        return 1.0 / (1 + np.exp(-x))

    def cost_func(self, y, pred):
        cost_1 = -y * np.log(pred)
        cost_0 = (1 - y) * np.log(1 - pred)
        cost = np.mean(cost_1 - cost_0)
        return cost

@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("-sep", "sep", default=",", help="csv separator")
@click.argument("model_file", default="model.json")
@click.option("-f", "features", multiple=True, help="select features")
@click.option("-p", "plot", is_flag=True, help="plot data")
@click.option("-e", "epochs", default=1, help="epochs to train")
@click.option("-l", "learning_rate", default=0.1, help="learning rate")
def main(data_file, features, sep, model_file, plot, epochs, learning_rate):
    trainer = Trainer(data_file, features, sep, model_file, plot, epochs, learning_rate)
    trainer.train()


if __name__ == "__main__":
    main()