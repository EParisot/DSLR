import matplotlib.pyplot as plt
from utils import read_data, get_numerics, get_classes, mean, std, clean, get_Y
import click
from math import isnan

@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("-sep", "sep", default=",", help="csv separator")
def main(data_file, sep):
    classes_column = "Hogwarts House"
    data = read_data(data_file, sep)
    num_data = get_numerics(data, False)
    Y = get_Y(data, classes_column)
    num_data, labels = clean(num_data, Y)
    class_list = get_classes(data, classes_column)
    fig = plt.figure("Scatter  plot")
    # for each matter
    data_cmp = []
    for j, key_a in enumerate(num_data.keys()):
        for i, key_b in enumerate(num_data.keys()):
            if i < j:
                ax = fig.add_subplot(len(num_data), len(num_data), len(num_data) * j + i + 1)
                if j == len(num_data)-1:
                    ax.set_xlabel(key_b, rotation=45)
                if i == 0:
                    ax.set_ylabel(key_a, rotation=45)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis='both', which='both', length=0)
                data_cmp.append((num_data[key_a], num_data[key_b]))
                classes = []
                # for each class
                for c in class_list:
                    c_tab_a = []
                    c_tab_b = []
                    # for each note 
                    for idx, label in enumerate(labels):
                        if label == c:
                            c_tab_a.append(float(num_data[key_a][idx]))
                            c_tab_b.append(float(num_data[key_b][idx]))
                    classes.append((c_tab_a, c_tab_b))
                for cat in classes:
                    ax.scatter(cat[0], cat[1], alpha=0.5)
    matter_1, matter_2 = find_similar_matters(num_data, data_cmp)
    print("Similar matters are %s and %s" % (matter_1, matter_2))
    plt.show(block = True)

def find_similar_matters(num_data, data_cmp):
    res = []
    for m_cmp in data_cmp:
        mean_diff_a = []
        mean_diff_b = []
        std_a = std(m_cmp[0])
        std_b = std(m_cmp[1])
        mean_a = mean(m_cmp[0])
        mean_b = mean(m_cmp[1])
        for i in m_cmp[0]:
            mean_diff_a.append(i - mean_a)
        for i in m_cmp[1]:
            mean_diff_b.append(i - mean_b)
        mean_diff = [a * b for a, b in list(zip(mean_diff_a, mean_diff_b))]
        cov_m_cmp = mean(mean_diff)
        res.append(cov_m_cmp / (std_a * std_b))
    k_max = 0
    id_max = 0
    for i, elem in enumerate(res):
        if abs(elem) > k_max:
            id_max = i
            k_max = abs(elem)    
    id = 0
    for j, key_a in enumerate(num_data.keys()):
        for i, key_b in enumerate(num_data.keys()):
            if i < j:
                if id == id_max:
                    return key_a, key_b
                id += 1

if __name__ == "__main__":
    main()