import matplotlib.pyplot as plt
from utils import read_data, get_numerics, get_classes
import click
import math

@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("-sep", "sep", default=",", help="csv separator")
def main(data_file, sep):
    classes_column = "Hogwarts House"
    data, _ = read_data(data_file, sep)
    num_data = get_numerics(data, False)
    class_list = get_classes(data, classes_column)
    fig = plt.figure("Pair plot")
    # for each matter
    for j, key_a in enumerate(num_data.keys()):
        for i, key_b in enumerate(num_data.keys()):
            ax = fig.add_subplot(len(num_data), len(num_data), len(num_data) * j + i + 1)
            if j == len(num_data)-1:
                ax.set_xlabel(key_b, rotation=45)
            if i == 0:
                ax.set_ylabel(key_a, rotation=45)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            if key_b != key_a:
                classes = []
                # for each class
                for c in class_list:
                    c_tab_a = []
                    c_tab_b = []
                    # for each note 
                    for idx, val in enumerate(data):
                        if val[classes_column] == c:
                            c_tab_a.append(float(num_data[key_a][idx]))
                            c_tab_b.append(float(num_data[key_b][idx]))
                    classes.append((c_tab_a, c_tab_b))
                for cat in classes:
                    ax.scatter(cat[0], cat[1], alpha=0.5)
            else:
                # for each class
                classes = []
                for c in class_list:
                    c_tab = []
                    # for each note
                    for idx, val in enumerate(num_data[key_a]):
                        if idx in class_list[c]:
                            c_tab.append(float(val))
                    classes.append(c_tab)
                for cat in classes:
                    clean_cat = []
                    for elem in cat:
                        if not math.isnan(elem):
                            clean_cat.append(elem)
                    ax.hist(clean_cat, alpha=0.5)
    plt.show(block = True)

if __name__ == "__main__":
    main()