import os
import re
import click
from utils import count, mean, std, _min, _max, q_1, q_2, q_3

class Descriptor(object):

    def __init__(self, data_file, sep):
        self.data_file = data_file
        self.sep = sep
        self.data = []
        self.labels = []
        self.features = []
        self.description = {}
        # Read data
        self.read_data()
        if len(self.data) == 0:
            print("Error : no valid data found in %s" % self.data_file)
            exit(0)

    def read_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file) as f:
                for i, line in enumerate(f):
                    line = line.replace('\n', '')
                    line_data = line.split(self.sep)
                    # read labels
                    if i == 0 and len(line_data) > 0:
                        for label in line_data:
                            self.labels.append(label)
                    # read data
                    elif len(line_data) > 0:
                        line_dict = {}
                        for j, feature in enumerate(line_data):
                            line_dict[self.labels[j]] = feature
                        self.data.append(line_dict)
    
    def descript(self):
        params = {
            "count": count,
            "mean ": mean,
            "std  ": std,
            "min  ": _min,
            "25%  ": q_1, 
            "50%  ": q_2, 
            "75%  ": q_3, 
            "max  ": _max
        }
        num_data = self.get_numerics()
        for feature in num_data:
            self.description[feature] = {}
            for param in params:
                self.description[feature][param] = params[param](num_data[feature])
        self.print_descript()

    def get_numerics(self):
        r = re.compile(r"-?\d+\.\d+")
        num_data = {}
        # double check num values
        for key in self.data[0]:
            if r.match(self.data[0][key]):
                num_data[key] = []
        for key in self.data[-1]:
            if r.match(self.data[-1][key]):
                num_data[key] = []
        # build numeric array
        for elem in self.data:
            for key in elem:
                if key in num_data:
                    if r.match(elem[key]):
                        num_data[key].append(float(elem[key]))
                    else:
                        num_data[key].append("NaN")
        return num_data

    def print_descript(self):
        def sep():
            for _ in range(lenght):
                print("_", end="")
            print()
        print("     ", end=" | ")
        lenght = 7
        for feature in self.description:
            print(feature, end=" | ")
            lenght += len(feature) + 3
        print()
        sep()
        for param in self.description[next(iter(self.description))]:
            print(param, end=" | ")
            for feature in self.description:
                sp_len = (len(feature) - len(str(round(self.description[feature][param], 2))))
                print(str(round(self.description[feature][param], 2)) + (" " * sp_len), end=" | ")
            print()
            sep()




@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("-sep", "sep", default=",", help="csv separator")
def main(data_file, sep):
    descriptor = Descriptor(data_file, sep)
    descriptor.descript()

if __name__ == "__main__":
    main()