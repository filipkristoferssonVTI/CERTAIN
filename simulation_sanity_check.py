import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from fpdf import fpdf


def create_pdf_from_folder(folder_name, save_path):
    pdf = fpdf.FPDF()
    image_files = sorted([f for f in os.listdir(folder_name) if f.endswith(".png")])

    if not image_files:
        print("No PNG images found in the folder.")
        return

    for image in image_files:
        image_path = os.path.join(folder_name, image)
        pdf.add_page(orientation='landscape')
        pdf.image(image_path)

    pdf.output(save_path)


def plot_HA_real_vs_simulation(HA, HA_simu, grid, save_path):
    HA_grouped = (grid
                  .merge(HA, on='C', how='left')
                  .groupby(['HA', 'C'])
                  .size()
                  .reset_index(name='occurrences'))

    HA_all = HA_grouped.merge(HA_simu[['C', 'HA', 'lambda_HA_C', 'N_HA_C']], how='outer', on=['C', 'HA'])
    HA_all = HA_all.fillna(0)

    unique_handelse = sorted(list(set(list(HA_grouped['HA'].unique()) + list(HA_simu['HA'].unique()))))

    for i, handelse in enumerate(unique_handelse):
        fig = go.Figure()

        HA = HA_all[HA_all['HA'] == handelse]

        fig.add_trace(
            go.Scatter(
                y=HA['N_HA_C'],
                x=HA['occurrences'],
                mode='markers',
                marker=dict(color='red', size=7),
                name=f'Real (sum={sum(HA["occurrences"])}) vs Simulated (sum={sum(HA["N_HA_C"])})'
            )
        )

        fig.add_trace(
            go.Scatter(
                y=HA['lambda_HA_C'],
                x=HA['occurrences'],
                mode='markers',
                marker=dict(color='blue', size=5),
                name=f'Real (sum={sum(HA["occurrences"])}) vs Estimated (sum={sum(HA["lambda_HA_C"])})'
            )
        )

        fig.update_layout(
            title_text=f"Occurrences Per Cell Real vs Simulated/Estimated - {handelse}",
            title_font=dict(size=14),
            showlegend=True,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
            )
        )

        fig.update_yaxes(title_text="Simulated/Estimated")
        fig.update_xaxes(title_text="Real")

        fig.write_image(save_path / f'{handelse}.png', engine='orca')


def main():
    data_folder = Path('data/output/eda')
    grid = pd.read_csv(data_folder / 'grid_mark.csv')
    HA = pd.read_csv(data_folder / 'ärenden.csv')
    HA_simu = pd.read_csv(data_folder / 'regression' / 'HA_for_plot.csv')

    HA_simu = HA_simu[HA_simu['N_HA_C'] != 0]

    grid = grid.rename(columns={'rut_id': 'C'})
    HA = HA.rename(columns={'handelse': 'HA', 'rut_id': 'C'})

    plot_HA_real_vs_simulation(HA, HA_simu, grid, data_folder / 'regression' / 'plots')
    create_pdf_from_folder(data_folder / 'regression' / 'plots',
                           data_folder / 'regression' / "HA_real_vs_HA_simu.pdf")


if __name__ == '__main__':
    main()
