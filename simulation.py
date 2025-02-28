from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson


def main():
    data_folder = Path('data/output/eda')

    # TODO: We skip the intercept coeff? Data is for (almost) three years?

    area_C_MA_all = pd.read_csv(data_folder / 'grid_mark.csv').set_index('rut_id').drop(columns='POP')
    beta_MA_HA_all = pd.read_csv(data_folder / 'regression' / 'coefficients.csv', index_col=0).drop(columns='intercept')

    # area_C_MA_all = area_C_MA_all / (1000 * 1000)  # TODO: Do we want this (normalize for cell size)?

    T_start = 0
    T_end = 3 * 365 * 24 * 60 * 60  # Approximate 3 years in seconds

    HA_realization = []
    for C, area_C_MA in area_C_MA_all.iterrows():
        for HA, beta_MA_HA in beta_MA_HA_all.iterrows():
            lambda_HA_C = sum(area_C_MA * beta_MA_HA)
            N_HA_C = poisson.rvs(lambda_HA_C)

            HA_start = np.random.default_rng().integers(T_start, T_end, size=N_HA_C)

            for start in HA_start:
                HA_realization.append({'C': C, 'HA': HA, 'start_time': start})

    result = pd.DataFrame(HA_realization).sort_values(by='start_time')
    result.to_csv(data_folder / 'regression' / 'HA_realization.csv')


if __name__ == '__main__':
    main()
