import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fpdf import fpdf
from scipy.stats import poisson


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


def plot_events(events, output_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, event_type in enumerate(events['Händelse, typ'].unique()):
            fig = go.Figure()

            e = events[events['Händelse, typ'] == event_type]

            fig.add_trace(
                go.Scatter(
                    y=e['N_HA_C'],
                    x=e['N_HA_C_real'],
                    mode='markers',
                    marker=dict(color='red', size=7),
                    name=f'Real (sum={round(sum(e["N_HA_C_real"]))}) vs '
                         f'Simulated (sum={round(sum(e["N_HA_C"]))})'
                )
            )

            fig.add_trace(
                go.Scatter(
                    y=e['lambda_HA_C'],
                    x=e['N_HA_C_real'],
                    mode='markers',
                    marker=dict(color='blue', size=5),
                    name=f'Real (sum={round(sum(e["N_HA_C_real"]))}) vs '
                         f'Estimated (sum={round(sum(e["lambda_HA_C"]), 1)})'
                )
            )

            fig.update_layout(
                title_text=f"Occurrences Per Cell Real vs Simulated/Estimated - {event_type}",
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

            fig.write_image(os.path.join(temp_dir, f'{event_type}.png'), engine='orca')

        create_pdf_from_folder(temp_dir, output_path)


def realize(estimated_simulated_events):
    T_start_s = 0
    T_end_s = 3 * 365 * 24 * 60 * 60  # Approximate 3 years in seconds

    HA_realization = []

    for _, row in estimated_simulated_events.iterrows():

        C = row['rut_id']
        HA = row['Händelse, typ']
        N_HA_C = row['N_HA_C']

        HA_start = np.random.default_rng().integers(T_start_s, T_end_s, size=N_HA_C)
        for start in HA_start:
            HA_realization.append({'rut_id': C, 'Händelse, typ': HA, 'start_time': start})

    return pd.DataFrame(HA_realization).sort_values(by='start_time')


def run_simulation():
    data_folder = Path('data/output_new')

    area_C_MA_all = pd.read_pickle(data_folder / 'grid.pkl').set_index('rut_id').drop(columns='geometry')
    beta_MA_HA_all = pd.read_csv(data_folder / 'coefficients.csv', index_col=0)

    area_C_MA_all = area_C_MA_all / (1000 * 1000)
    area_C_MA_all = area_C_MA_all.assign(intercept=1)

    assert list(area_C_MA_all.columns) == list(beta_MA_HA_all.columns), 'Datasets needs to have the same MA cols.'

    estimated_simulated_events = []

    for C, area_C_MA in area_C_MA_all.iterrows():
        for HA, beta_MA_HA in beta_MA_HA_all.iterrows():
            lambda_HA_C = np.exp(area_C_MA.values @ beta_MA_HA.values)
            N_HA_C = poisson.rvs(lambda_HA_C)

            estimated_simulated_events.append({
                'rut_id': C,
                'Händelse, typ': HA,  # type of event
                'lambda_HA_C': lambda_HA_C,  # estimated nr of events
                'N_HA_C': N_HA_C  # simulated nr of events
            })

    events = pd.DataFrame(estimated_simulated_events)

    real_events = (pd.read_pickle(data_folder / 'events.pkl')[['Ärende, årsnr', 'rut_id', 'Händelse, typ']]
                   .drop_duplicates(subset='Ärende, årsnr')
                   .drop(columns='Ärende, årsnr')
                   .groupby(['rut_id', 'Händelse, typ'])
                   .size()
                   .reset_index(name='N_HA_C_real'))

    events = events.merge(real_events, on=['rut_id', 'Händelse, typ'], how='outer').fillna(0)

    events.to_csv(data_folder / 'estimated_simulated_events.csv')


def main():
    # run_simulation()

    data_folder = Path('data/output_new')
    events = pd.read_csv(data_folder / 'estimated_simulated_events.csv')

    plot_events(events, data_folder / 'event_plots.pdf')


if __name__ == '__main__':
    main()
