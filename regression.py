import os
from pathlib import Path

import fpdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import lstsq


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
    slope, _ = np.polyfit(y, y_pred, 1)

    plt.scatter(y, y_pred, color="blue", alpha=0.6)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs. Predicted Values - {handelse_id}")

    plt.plot(y, slope * y, color="red", linewidth=2, label=f"Trendline: y = {slope}x")
    plt.plot([min(y), max(y)], [min(y), max(y)], color="green", label="Ideal: y = x")

    plt.legend()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(df, save_path):
    plt.figure(figsize=(15, 10))
    sns.heatmap(df, cmap="coolwarm", yticklabels=df.index)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches="tight")
    plt.close()


def main():
    data_folder = Path('data/output/eda')
    output_folder = data_folder / 'regression'

    handelse = pd.read_csv(data_folder / 'ärenden.csv')
    grid = pd.read_csv(data_folder / 'grid_mark.csv')

    coeffs = {}

    for single_handelse in handelse['handelse'].dropna().unique():
        handelse_count = handelse.loc[handelse['handelse'] == single_handelse].groupby('rut_id').count().reset_index()
        g = grid.merge(handelse_count, how='left', on='rut_id').fillna(0)
        y = g['handelse'].values
        design_matrix = g.drop(columns=['handelse', 'rut_id', 'POP']).values

        coefficients, residuals, rank, s = lstsq(design_matrix, y)
        y_pred = design_matrix @ coefficients
        coeffs[single_handelse] = coefficients

        save_regression_plot(y, y_pred, single_handelse, output_folder / single_handelse)

    create_pdf_from_folder(output_folder, output_folder / 'regression_test.pdf')

    heatmap_df = pd.DataFrame(coeffs.values(),
                              columns=grid.drop(columns=['rut_id', 'POP']).columns,
                              index=[handelse_id for handelse_id in coeffs.keys()])
    # plot_heatmap(heatmap_df, output_folder / 'heatmap')


if __name__ == '__main__':
    main()
