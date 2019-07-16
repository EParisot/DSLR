import os
import click
import matplotlib.pyplot as plt
import random
from utils import read_data, read_model, save_model, get_numerics, get_Y, _min, _max

class Trainer(object):

    def __init__(self, data_file, features, sep, model_file, plot, epochs, lr):
        self.model_file = model_file
        self.features = features
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
        # Read model
        if len(model_file):
            self.model, _ = read_model(model_file)
        
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
            self.ranges[key] = {}
            k_min = _min(X[key])
            k_max = _max(X[key])
            self.ranges[key]["min"] = k_min
            self.ranges[key]["max"] = k_max
            for val in X[key]:
                norm_X[key].append((val - k_min) / (k_max - k_min))
        return norm_X

    def train(self):
        X = get_numerics(self.data, False)
        # select features:
        if len(self.features) == 0:
            self.features = []
            for key in X.keys():
                self.features.append(key)
        X_tmp = {}
        for feat in X:
            if feat in self.features:
                X_tmp[feat] = X[feat]
        X = X_tmp
        # clean and normalise
        clean_X = self.clean(X)
        norm_X = self.normalise(clean_X)
        classes_column = "Hogwarts House"
        Y = get_Y(self.data, classes_column)
        # train
        self.train_loop(norm_X, Y)
        # save model
        save_model(self.model, self.ranges, self.model_file)
        # plot result
        if self.plot:
            plt.figure("Train history")
            plt.plot(self.acc, label="acc")
            plt.plot(self.loss, label="loss")
            plt.legend()
            plt.show(block=True)
    
    def train_loop(self, X, Y):
        # loop on epochs / batches / data_points
        for epoch in range(self.epochs):
            pass

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