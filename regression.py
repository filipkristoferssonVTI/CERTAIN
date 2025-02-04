from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import lstsq


def main():
    data_folder = Path('data/output/eda')
    handelse = pd.read_csv(data_folder / 'ärenden.csv')
    grid = pd.read_csv(data_folder / 'grid_mark.csv')

    handelse_1 = handelse.loc[handelse['handelse'] == 'Brand eller brandtillbud i byggnad']
    handelse_1 = handelse_1.groupby('rut_id').count().reset_index()

    grid = grid.merge(handelse_1, how='left', on='rut_id').fillna(0)

    target = grid['handelse'].values
    design_matrix = grid.drop(columns=['handelse', 'rut_id', 'POP']).values

    coefficients, residuals, rank, s = lstsq(design_matrix, target)

    # Optional: Scatter plot of actual vs predicted values
    y_pred = design_matrix @ coefficients
    plt.scatter(target, y_pred, color="blue", alpha=0.6)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values (Multivariate Regression)")
    plt.plot([min(target), max(target)], [min(target), max(target)], color="red", linestyle="--")  # Diagonal line
    plt.show()

    DEBUG = 1


if __name__ == '__main__':
    main()
