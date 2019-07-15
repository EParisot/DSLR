import matplotlib.pyplot as plt
from utils import read_data, get_numerics, get_classes
import click

@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("-sep", "sep", default=",", help="csv separator")
def main(data_file, sep):
    pass

if __name__ == "__main__":
    main()