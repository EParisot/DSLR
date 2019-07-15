import matplotlib.pyplot as plt
from utils import read_data, get_numerics, get_classes
import click

@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("-sep", "sep", default=",", help="csv separator")
def main(data_file, sep):
    data = read_data(data_file, sep)
    num_data = get_numerics(data, True)
    classes = get_classes(data, "Hogwarts House")
    fig = plt.figure()
    # for each matter
    for i, key in enumerate(num_data.keys()):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.set_title(key)
        # for each class
        for c in classes:
            c_tab = []
            # for each note
            for j, val in enumerate(num_data[key]):
                if j in classes[c]:
                    c_tab.append(val)
            ax.hist(c_tab, alpha=0.5)
    fig.legend(classes.keys(), loc = (0.8, 0))
    fig.tight_layout()
    plt.show(block = True)

if __name__ == "__main__":
    main()