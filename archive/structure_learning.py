from pathlib import Path

import pandas as pd
from pgmpy.estimators import HillClimbSearch

from bayesian_network import filter_data
from utils.model import plot_model


def main():
    data_folder = Path('../data')
    df = pd.read_csv(data_folder / 'Kopia av Daedalos export - Insatta resurser 2201 2411.csv', sep=';')

    df = filter_data(df)

    est = HillClimbSearch(df)
    dag = est.estimate(scoring_method='k2score')

    plot_model(dag)


if __name__ == '__main__':
    main()
