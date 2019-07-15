import matplotlib.pyplot as plt
from utils import read_data, get_numerics, get_classes
import click

@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("-sep", "sep", default=",", help="csv separator")
def main(data_file, sep):
    classes_column = "Hogwarts House"
    data = read_data(data_file, sep)
    num_data = get_numerics(data, False)
    class_list = get_classes(data, classes_column)
    fig = plt.figure()
    # for each matter
    for j, key_a in enumerate(num_data.keys()):
        for i, key_b in enumerate(num_data.keys()):
            if key_b != key_a:
                ax = fig.add_subplot(len(num_data), len(num_data), len(num_data) * j + i + 1)
                #ax.set_title(key_a + "/" + key_b)
                ax.axis("off")
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
    fig.tight_layout()
    plt.show(block = True)

if __name__ == "__main__":
    main()