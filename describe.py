import os
import click
from utils import read_data, get_numerics, count, mean, std, _min, _max, q_1, q_2, q_3

class Descriptor(object):

    def __init__(self, data_file, sep):
        self.data_file = data_file
        self.sep = sep
        self.features = []
        self.description = {}
        # Read data
        self.data, _ = read_data(self.data_file, self.sep)
        if len(self.data) == 0:
            print("Error : no valid data found in %s" % self.data_file)
            exit(0)
    
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
        num_data = get_numerics(self.data, False)
        for feature in num_data:
            self.description[feature] = {}
            for param in params:
                self.description[feature][param] = params[param](num_data[feature])
        self.print_descript()

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