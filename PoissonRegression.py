import os
from pathlib import Path

import fpdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.optimize as opt
import scipy.stats as stats
from scipy.linalg import lstsq
from statsmodels.compat import scipy


def create_pdf_from_folder(folder_name, save_path):
    pdf = fpdf.FPDF()
    image_files = sorted([f for f in os.listdir(folder_name) if f.endswith(".png")])

    if not image_files:
        print("No PNG images found in the folder.")
        return

    for image in image_files:
        image_path = os.path.join(folder_name, image)
        pdf.add_page()
        pdf.image(image_path, x=10, y=10, w=180)

    pdf.output(save_path)


def r2(y, y_pred):
    ss_total = np.sum((y - np.mean(y)) ** 2)  # total sum of squares
    ss_residual = np.sum((y - y_pred) ** 2)  # residual sum of squares
    return 1 - (ss_residual / ss_total)


def save_regression_plot(y, y_pred, handelse_id, save_path):

    plt.scatter(y, y_pred, color="blue", alpha=0.6)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs. Predicted Values - {handelse_id}")

    slope, _ = np.polyfit(y, y_pred, 1)
    plt.plot(y, slope * y, color="red", linewidth=2, label=f"Trendline: y = {slope}x")
    # plt.plot([min(y), max(y)], [min(y), max(y)], color="green", label="Ideal: y = x")

    plt.legend()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches="tight")
    plt.close()

def print_confidence(y, y_pred, confidence):

    conf_intervals = np.array([stats.poisson.interval(confidence, m) for m in y_pred])
    # print(conf_intervals)
    # input("Waiting")

    within_band = (y >= conf_intervals[:,0]) & (y <= conf_intervals[:,1])
    oneormore = (y_pred >= 1)
    within_band_and_oneormore = (within_band & oneormore)
    print("    ", np.round(100 * np.sum(within_band) / y.shape[0], 2), "% within ", confidence, "-confidence")
    print("    ", np.round(np.sum(oneormore), 2), "observations >=1")
    print("    ", np.round(100 * np.sum(within_band_and_oneormore) / np.sum(oneormore), 2), "% of >=1 observations within confidence")


def plot_heatmap(df, save_path):
    plt.figure(figsize=(15, 10))
    sns.heatmap(df, cmap="coolwarm", yticklabels=df.index)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches="tight")
    plt.close()

# Define the Poisson negative log-likelihood
def poisson_neg_log_likelihood(beta, X, y):
    linear_pred = X @ beta
    expected_counts = np.exp(linear_pred)  # Poisson mean
    return -np.sum(y * linear_pred - expected_counts)  # Negative log-likelihood

def main():
    data_folder = Path('data/output/eda')
    output_folder = data_folder / 'regression'

    handelse = pd.read_csv(data_folder / 'ärenden.csv')
    grid = pd.read_csv(data_folder / 'grid_mark.csv')

    coeffs = {}

    for single_handelse in handelse['handelse'].dropna().unique():

        print(single_handelse)

        handelse_count = handelse.loc[handelse['handelse'] == single_handelse].groupby('rut_id').count().reset_index()
        g = grid.merge(handelse_count, how='left', on='rut_id').fillna(0)
        y = g['handelse'].values
        design_matrix = g.drop(columns=['handelse', 'rut_id', 'POP']).values
        design_matrix = np.column_stack((design_matrix / 1e6, np.ones(design_matrix.shape[0])))

        n_samples, n_features = design_matrix.shape
        # print(n_samples)
        # print(n_features)
        #input("Pausing...")

        beta_init = np.zeros(n_features)
        result = opt.minimize(poisson_neg_log_likelihood, beta_init, args=(design_matrix, y), method="L-BFGS-B")
        coefficients = result.x
        print("... ", result.message)
        # print(coefficients.shape)
        # input("Waitin")
        # coefficients, residuals, rank, s = lstsq(design_matrix, y)

        linear_pred = design_matrix @ coefficients
        y_pred = np.exp(linear_pred)  # Poisson mean
        coeffs[single_handelse] = coefficients

        print_confidence(y, y_pred, 0.95)

        save_regression_plot(y, y_pred, single_handelse, output_folder / single_handelse)

    print("Done with estimation, about to create pdf")

    create_pdf_from_folder(output_folder, output_folder / 'regression_test.pdf')

    print("Done with pdf, about to create heatmap")

    heatmap_df = pd.DataFrame(coeffs.values(),
                              columns=grid.drop(columns=['rut_id', 'POP']).columns,
                              index=[handelse_id for handelse_id in coeffs.keys()])
    # plot_heatmap(heatmap_df, output_folder / 'heatmap')


if __name__ == '__main__':
    main()
