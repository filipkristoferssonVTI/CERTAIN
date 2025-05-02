from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats


def print_confidence(y, y_pred, confidence):
    conf_intervals = np.array([stats.poisson.interval(confidence, m) for m in y_pred])

    within_band = (y >= conf_intervals[:, 0]) & (y <= conf_intervals[:, 1])
    oneormore = (y_pred >= 1)
    within_band_and_oneormore = (within_band & oneormore)
    print("    ", np.round(100 * np.sum(within_band) / y.shape[0], 2), "% within ", confidence, "-confidence")
    print("    ", np.round(np.sum(oneormore), 2), "observations >=1")
    print("    ", np.round(100 * np.sum(within_band_and_oneormore) / np.sum(oneormore), 2),
          "% of >=1 observations within confidence")


def poisson_neg_log_likelihood(beta, X, y):
    """Define the Poisson negative log-likelihood."""
    linear_pred = X @ beta
    expected_counts = np.exp(linear_pred)  # Poisson mean
    return -np.sum(y * linear_pred - expected_counts)  # negative log-likelihood


def generate_coefficients():
    data_folder = Path('data/output')

    events = pd.read_pickle(data_folder / 'processed_events.pkl')
    grid = pd.read_pickle(data_folder / 'grid.pkl')

    events = events.drop_duplicates(subset='Ärende, årsnr')  # to not count the same event more than once.

    design_matrix_cols = [col for col in grid.columns if col not in ['rut_id', 'geometry']]

    area_threshold = 1e6
    below_threshold = [col for col in design_matrix_cols if grid[col].sum() < area_threshold]
    print(f"Columns with sum less than {area_threshold}: {below_threshold}, these are dropped.")
    design_matrix_cols = [col for col in design_matrix_cols if col not in below_threshold]

    coeffs = {}

    for event_type in events['Händelse, typ'].dropna().unique():
        event_count = (events[['rut_id', 'Händelse, typ']]
                       .loc[events['Händelse, typ'] == event_type]
                       .groupby('rut_id')
                       .count()
                       .reset_index())

        g = grid.merge(event_count, how='left', on='rut_id').fillna(0)
        y = g['Händelse, typ'].values

        design_matrix = g[design_matrix_cols].values
        # An intercept columns is added with ones
        design_matrix = np.column_stack((design_matrix / 1e6, np.ones(design_matrix.shape[0])))

        n_samples, n_features = design_matrix.shape

        beta_init = np.zeros(n_features)
        result = opt.minimize(poisson_neg_log_likelihood, beta_init, args=(design_matrix, y), method="L-BFGS-B")
        coefficients = result.x

        print("... ", result.message)

        linear_pred = design_matrix @ coefficients
        y_pred = np.exp(linear_pred)  # Poisson mean
        coeffs[event_type] = coefficients

        print_confidence(y, y_pred, 0.95)

    design_matrix_cols.append('intercept')

    (pd.DataFrame(coeffs.values(),
                  columns=design_matrix_cols,
                  index=[event_type for event_type in coeffs.keys()])
     .to_csv(data_folder / 'coefficients.csv'))


def generate_lambda_vals():
    data_folder = Path('data/output')

    area_C_MA_all = pd.read_pickle(data_folder / 'grid.pkl').set_index('rut_id').drop(columns='geometry')
    beta_MA_HA_all = pd.read_csv(data_folder / 'coefficients.csv', index_col=0)

    area_C_MA_all = area_C_MA_all / (1000 * 1000)
    area_C_MA_all = area_C_MA_all.assign(intercept=1)
    area_C_MA_all = area_C_MA_all[[col for col in beta_MA_HA_all.columns]]

    assert list(area_C_MA_all.columns) == list(beta_MA_HA_all.columns), 'Datasets needs to have the same MA cols.'

    lambda_vals = []

    for C, area_C_MA in area_C_MA_all.iterrows():
        for HA, beta_MA_HA in beta_MA_HA_all.iterrows():
            lambda_HA_C = np.exp(area_C_MA.values @ beta_MA_HA.values)

            lambda_vals.append({
                'cell_id': C,
                'event_type': HA,
                'lambda': lambda_HA_C,  # estimated nr of events
            })

    lambda_vals = pd.DataFrame(lambda_vals)

    lambda_vals.to_csv(data_folder / 'lambda_vals.csv', index=False)


def main():
    # generate_coefficients()
    generate_lambda_vals()


if __name__ == '__main__':
    main()
