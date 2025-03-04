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
        pdf.image(image_path,  # x=10, y=10, w=180
                  )

    pdf.output(save_path)


def plot_handelse_population(HA, HA_realization, grid, save_path):
    HA_grouped = (grid
                  .merge(HA, on='C', how='left')
                  .groupby(['HA', 'C', 'POP'])
                  .size()
                  .reset_index(name='occurrences'))

    HA_realization_grouped = (grid
                              .merge(HA_realization, on='C', how='left')
                              .groupby(['HA', 'C', 'POP'])
                              .size()
                              .reset_index(name='occurrences'))

    unique_handelse = sorted(list(set(list(HA_grouped['HA'].unique()) + list(HA_realization_grouped['HA'].unique()))))

    for i, handelse in enumerate(unique_handelse):
        fig = go.Figure()

        # HA real
        real = HA_grouped[HA_grouped['HA'] == handelse]
        fig.add_trace(
            go.Scatter(
                x=real['POP'],
                y=real['occurrences'],
                mode='markers',
                marker=dict(color='blue', size=6),
                name=f'Real - n={sum(real["occurrences"])}')
        )
        # HA simulated
        simulated = HA_realization_grouped[HA_realization_grouped['HA'] == handelse]
        fig.add_trace(
            go.Scatter(
                x=simulated['POP'],
                y=simulated['occurrences'],
                mode='markers',
                marker=dict(color='red', size=6),
                name=f'Simulated - n={sum(simulated["occurrences"])}')
        )

        fig.update_layout(
            title_text=f"Occurrences Per Cell by Population - {handelse}",
            title_font=dict(size=14),
            showlegend=True,
            template="plotly_white",
        )

        fig.update_xaxes(title_text="Population")

        fig.write_image(save_path / f'{handelse}.png', engine='orca')


def main():
    data_folder = Path('data/output/eda')
    grid = pd.read_csv(data_folder / 'grid_mark.csv')
    HA = pd.read_csv(data_folder / 'ärenden.csv')
    HA_realization = pd.read_csv(data_folder / 'regression' / 'HA_realization.csv')

    grid = grid.rename(columns={'rut_id': 'C'})
    HA = HA.rename(columns={'handelse': 'HA', 'rut_id': 'C'})

    plot_handelse_population(HA, HA_realization, grid, data_folder / 'regression' / 'plots')
    create_pdf_from_folder(data_folder / 'regression' / 'plots',
                           data_folder / 'regression' / "HA_vs_HA_realization.pdf")


if __name__ == '__main__':
    main()
