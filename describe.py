import os
import re
import click

class Descriptor(object):

    def __init__(self, data_file, sep):
        self.data_file = data_file
        self.sep = sep
        self.data = []
        self.labels = []
        # Read data
        self.read_data()
        if len(self.data) == 0:
            print("Error : no valid data found in %s" % self.data_file)
            return
        print(self.data[42])

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
                        data = {}
                        for j, feature in enumerate(line_data):
                            data[self.labels[j]] = feature
                            self.data.append(data)
    
    def descript(self):
        pass

@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("-sep", "sep", default=",", help="csv separator")
def main(data_file, sep):
    descriptor = Descriptor(data_file, sep)
    descriptor.descript()

if __name__ == "__main__":
    main()