from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson


def main():
    data_folder = Path('data/output/eda')

    area_C_MA_all = pd.read_csv(data_folder / 'grid_mark.csv').set_index('rut_id').drop(columns='POP')
    beta_MA_HA_all = pd.read_csv(data_folder / 'regression' / 'coefficients.csv', index_col=0)

    area_C_MA_all = area_C_MA_all / (1000 * 1000)
    area_C_MA_all = area_C_MA_all.assign(intercept=1)

    T_start_s = 0
    T_end_s = 3 * 365 * 24 * 60 * 60  # Approximate 3 years in seconds

    assert list(area_C_MA_all.columns) == list(beta_MA_HA_all.columns), 'Datasets needs to have the same MA cols.'
    
    HA_realization = []
    for C, area_C_MA in area_C_MA_all.iterrows():
        for HA, beta_MA_HA in beta_MA_HA_all.iterrows():
            lambda_HA_C = np.exp(area_C_MA.values @ beta_MA_HA.values)
            N_HA_C = poisson.rvs(lambda_HA_C)

            HA_start = np.random.default_rng().integers(T_start_s, T_end_s, size=N_HA_C)

            for start in HA_start:
                HA_realization.append({'C': C, 'HA': HA, 'start_time': start})

    result = pd.DataFrame(HA_realization).sort_values(by='start_time')
    result.to_csv(data_folder / 'regression' / 'HA_realization.csv')


if __name__ == '__main__':
    main()
