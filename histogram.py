import matplotlib.pyplot as plt
from utils import read_data, get_numerics, get_classes, mean, std, _min
import click
import math

@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("-sep", "sep", default=",", help="csv separator")
def main(data_file, sep):
    classes_column = "Hogwarts House"
    data, _ = read_data(data_file, sep)
    num_data = get_numerics(data)
    class_list = get_classes(data, classes_column)

    matter = find_most_equal(num_data, class_list)
    print("Matter with the most homogeneous repartition between houses : %s" % matter)

    fig = plt.figure("Histogram")
    # for each matter
    for i, key in enumerate(num_data.keys()):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.set_title(key)
        classes = []
        # for each class
        for c in class_list:
            c_tab = class_tab(num_data[key], class_list[c])
            classes.append(c_tab)
        for cat in classes:
            clean_cat = []
            for elem in cat:
                if not math.isnan(elem):
                    clean_cat.append(elem)
            ax.hist(clean_cat, alpha=0.5)
    fig.legend(class_list.keys(), loc = (0.8, 0))
    fig.tight_layout()
    plt.show(block = True)

def class_tab(num_list, _class):
    c_tab = []
    # for each note
    for j, val in enumerate(num_list):
        if j in _class:
            c_tab.append(float(val))
    return c_tab

def find_most_equal(data, class_list):
    m_std = {}
    for matter in data.keys():
        m_stds = []
        for _class in class_list:
            c_tab = class_tab(data[matter], class_list[_class])
            m_stds.append(std(c_tab))
        m_std[matter] = std(m_stds)
    return min(m_std, key=m_std.get)

if __name__ == "__main__":
    main()
