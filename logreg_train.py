import os
import click
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import read_data, read_model, save_model, get_numerics, clean, classes_list, get_Y, _min, _max

class Trainer(object):

    def __init__(self, data_file, features, sep, model_file, plot, epochs, lr):
        self.model_file = model_file
        self.features = list(features)
        self.plot = plot
        self.epochs = epochs
        self.model = {}
        self.ranges = {}
        self.lr = lr
        self.lr_hist = []
        self.acc = []
        self.loss = []
        # Read data
        self.data = read_data(data_file, sep)
        if len(self.data) == 0:
            print("Error : no valid data found in %s" % data_file)
            exit(0)
        self.classes_column = "Hogwarts House"
        self.classes = classes_list(self.data, self.classes_column)
        # Read model
        if len(model_file):
            self.model, _, _ = read_model(model_file, self.classes)

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
        # buid Y
        Y = get_Y(self.data, self.classes_column)
        # select X features:
        X = self.select_feat()
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
        return np_X, clean_Y

    def train(self):
        #preprocess data
        X, Y = self.preprocess()
        # one vs all
        for curr_class in self.classes:
            print("Training on %s" % curr_class)
            # build tmp Y (one v all)
            tmp_Y = np.zeros((len(Y), 1))
            for i, val in enumerate(Y):
                if val == curr_class:
                    tmp_Y[i] = 1
            # build thetas np array
            thetas = np.zeros((len(self.features), 1))
            for i, theta in enumerate(self.model[curr_class]):
                thetas[i] = self.model[curr_class][theta]
            # train
            loss, acc, lr = self.train_class(X, tmp_Y, thetas, curr_class, self.lr)
            self.acc.append(acc)
            self.loss.append(loss)
            self.lr_hist.append(lr)
            # save model
            save_model(self.model, self.ranges, self.model_file)
        # plot result
        if self.plot:
            plt.figure("Train history")
            for i, acc in enumerate(self.acc):
                plt.plot(acc, label="acc_" + self.classes[i])
            for i, loss in enumerate(self.loss):
                plt.plot(loss, label="loss_" + self.classes[i])
            plt.legend()
            plt.show(block=True)
    
    def train_class(self, X, Y, thetas, curr_class, lr):
        loss_class = []
        acc_class = []
        lr_class = []
        for epoch in range(self.epochs):
            print("Epoch : %d" % (epoch + 1))
            # process train epoch
            loss, acc, lr = self.train_epoch(X, Y, thetas, curr_class, loss_class, lr)
            print("loss : %f ; acc : %f" % (round(loss, 2), round(acc, 2)))
            loss_class.append(loss)
            acc_class.append(acc)
            lr_class.append(lr)
            # Dynamic plot
            if self.plot:
                self.animate()
        return loss_class, acc_class, lr_class

    def animate(self):
        plt.clf()
        # TODO Plot scatter and loss curve
        
        # plot learning rate history
        plt.twinx().twiny()
        for lr in self.lr_hist:
            plt.plot(lr, label="Learning Rate")
        if len(self.lr_hist):
            plt.legend()
        plt.draw()
        plt.pause(1/self.epochs)
    
    def train_epoch(self, X, Y, thetas, curr_class, loss_class, lr_class):
        # train
        pred = self.predict(np.dot(X, thetas))
        loss = self.cost_func(Y, pred)
        thetas -= (lr_class * np.dot(X.T, (pred - Y)))
        # save thetas
        for i, feature in enumerate(self.features):
            self.model[curr_class][feature] = thetas[i][0]
        # adjust lr
        if len(loss_class) > 0:
            if loss >= loss_class[-1]:
                for i, feature in enumerate(self.features):
                    self.model[curr_class][feature] += lr_class * thetas[i] / len(Y)
                lr_class *=  0.5
            else:
                lr_class *= 1.05
        # metrics
        new_pred = self.predict(np.dot(X, thetas))
        acc = np.mean(1 - (Y - new_pred))
        return loss, acc, lr_class

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