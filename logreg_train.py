import os
import click
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import read_data, read_model, save_model, get_numerics, hard_clean, classes_list, get_Y, _min, _max

class Trainer(object):

    def __init__(self, data_file, features_file, sep, model_file, plot, epochs, lr):
        self.model_file = model_file
        self.plot = plot
        self.epochs = epochs
        self.model = {}
        self.ranges = {}
        self.lr = lr
        self.acc = []
        self.loss = []
        self.val_acc = []
        self.val_loss = []
        # Read data
        self.data, self.labels = read_data(data_file, sep)
        if len(self.data) == 0:
            print("Error : no valid data found in %s" % data_file)
            exit(0)
        self.features = self.get_features(features_file)
        self.classes_column = "Hogwarts House"
        self.classes = classes_list(self.data, self.classes_column)
        # Read model
        if len(model_file):
            self.model, _, _ = read_model(model_file, self.classes)
        self.X_train = []
        self.X_val = []
        self.Y_train = []
        self.Y_val = []
        self.thetas = []
        self.curr_class = ""

    def get_features(self, features_file):
        features = []
        if len(features_file) > 0 and os.path.exists(features_file):
            with open(features_file) as f:
                for line in f:
                    feature = line.replace("\n", "")
                    if len(feature) > 0 and feature in self.labels:
                        features.append(feature)
        else:
            print("No features file")
            exit(0)
        return features

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
        X = get_numerics(self.data, get_hand=True)
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
        # clean and normalise X
        clean_X, clean_Y = hard_clean(X, Y)
        norm_X = self.normalise(clean_X)
        # cast X to np.array
        np_X = np.empty((len(clean_Y), len(norm_X)))
        for i, key in enumerate(self.features):
            for idx, _ in enumerate(clean_Y):
                np_X[idx][i] = norm_X[key][idx]
        # Shuffle data
        l = list(zip(np_X, clean_Y))
        random.shuffle(l)
        X, Y = zip(*l)
        # Validation split
        split_point = int(0.8 * len(X))
        self.X_train = np.array(X[:split_point])
        self.Y_train = np.array(Y[:split_point])
        self.X_val = np.array(X[split_point:])
        self.Y_val = np.array(Y[split_point:])

    def train(self):
        #preprocess data
        self.preprocess()
        # one vs all
        if self.plot == True:
            plt.figure("Classes")
        for self.curr_class in self.classes:
            print("Training on %s" % self.curr_class)
            # build tmp Y (one v all)
            tmp_Y = np.zeros((len(self.Y_train), 1))
            for i, val in enumerate(self.Y_train):
                if val == self.curr_class:
                    tmp_Y[i] = 1
            tmp_Y_val = np.zeros((len(self.Y_val), 1))
            for i, val in enumerate(self.Y_val):
                if val == self.curr_class:
                    tmp_Y_val[i] = 1
            # build thetas np array
            self.thetas = np.zeros((len(self.features), 1))
            for i, self.theta in enumerate(self.model[self.curr_class]):
                self.thetas[i] = self.model[self.curr_class][self.theta]
            # train
            loss, acc, val_loss, val_acc = self.train_class(tmp_Y, tmp_Y_val)
            
            print("loss : %f ; acc : %f" % (round(loss[-1], 2), round(acc[-1], 2)), flush=True)
            print("val_loss : %f ; val_acc : %f" % (round(val_loss[-1], 2), round(val_acc[-1], 2)),  flush=True)
            

            self.acc.append(acc)
            self.loss.append(loss)
            self.val_acc.append(val_acc)
            self.val_loss.append(val_loss)
            if self.plot == True:
                self.animate(tmp_Y)
            # save model
            save_model(self.model, self.ranges, self.model_file)
        # plot result
        if self.plot:
            plt.figure("History")
            plt.subplot(1, 2, 1)
            plt.title("Train history")
            for i, acc in enumerate(self.acc):
                plt.plot(acc, label="acc_" + self.classes[i])
            for i, loss in enumerate(self.loss):
                plt.plot(loss, label="loss_" + self.classes[i])
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.title("Validation history")
            for i, val_acc in enumerate(self.val_acc):
                plt.plot(val_acc, label="val_acc_" + self.classes[i])
            for i, val_loss in enumerate(self.val_loss):
                plt.plot(val_loss, label="val_loss_" + self.classes[i])
            plt.legend()
            plt.show()
    
    def train_class(self, Y, Y_val):
        loss_class = []
        acc_class = []
        val_loss_class = []
        val_acc_class = []
        for epoch in range(self.epochs):
            # process train epoch
            loss, acc, val_loss, val_acc = self.train_epoch(Y, Y_val, loss_class)
            # Check result
            if epoch and loss_class[-1] - loss < 0.000001:
                if self.plot == True:
                    self.animate(Y)
                return loss_class, acc_class, val_loss_class, val_acc_class
            loss_class.append(loss)
            acc_class.append(acc)
            val_loss_class.append(val_loss)
            val_acc_class.append(val_acc)
        return loss_class, acc_class, val_loss_class, val_acc_class

    def animate(self, Y):
        plt.subplot(2, 2, self.classes.index(self.curr_class)+1)
        plt.title(self.curr_class)
        plt.scatter(range(len(Y)), sorted(Y))
        plt.twinx().twiny()
        pred = self.predict(np.dot(self.X_train, self.thetas))
        plt.plot(sorted(pred))
        plt.tight_layout()
        plt.draw()
    
    def train_epoch(self, Y, Y_val, loss_class):
        # train
        pred = self.predict(np.dot(self.X_train, self.thetas))
        gradient = np.dot(self.X_train.T, (pred - Y)) / len(Y)
        self.thetas -= self.lr * gradient
        # save thetas
        for i, feature in enumerate(self.features):
            self.model[self.curr_class][feature] = self.thetas[i][0]
        # metrics
        new_pred = self.predict(np.dot(self.X_train, self.thetas))
        loss = self.cost_func(Y, new_pred)
        acc = np.mean(1 - abs(Y - new_pred))
        # Cross validation
        val_pred = self.predict(np.dot(self.X_val, self.thetas))
        val_loss = self.cost_func(Y_val, val_pred)
        val_acc = np.mean(1 - abs(Y_val - val_pred))
        return loss, acc, val_loss, val_acc

    def predict(self, x):
        return 1.0 / (1 + np.exp(-x))

    def cost_func(self, y, pred):
        cost_1 = y * np.log(pred)
        cost_0 = (1 - y) * np.log(1 - pred)
        cost = -np.sum(cost_1 + cost_0) / len(y)
        return cost

@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("-sep", "sep", default=",", help="csv separator")
@click.argument("model_file", default="model.json")
@click.option("-f", "features_file", default="features.csv", help="selected features file (one by line)")
@click.option("-p", "plot", is_flag=True, help="plot data")
@click.option("-e", "epochs", default=1, help="epochs to train")
@click.option("-l", "learning_rate", default=0.1, help="learning rate")
def main(data_file, features_file, sep, model_file, plot, epochs, learning_rate):
    trainer = Trainer(data_file, features_file, sep, model_file, plot, epochs, learning_rate)
    trainer.train()


if __name__ == "__main__":
    main()
