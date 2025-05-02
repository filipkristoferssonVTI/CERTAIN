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


def create_qgis_data():
    data_folder = Path('data/output')
    events = pd.read_csv(data_folder / 'estimated_simulated_events.csv', dtype={'rut_id': str})
    grid = pd.read_pickle(data_folder / 'grid.pkl')

    HA = ['Hjärtstopp', 'Brand eller brandtillbud i byggnad', 'Brand eller brandtillbud i skog eller mark',
          'Drunkning eller drunkningstillbud']

    grid = grid[['geometry', 'rut_id']]

    events_real = events.loc[events['N_HA_C_real'] != 0]
    events_simu = events.loc[events['N_HA_C'] != 0]

    events_real = grid.merge(events_real, how='left', on='rut_id')
    events_simu = grid.merge(events_simu, how='left', on='rut_id')

    events_real = events_real[['rut_id', 'geometry', 'N_HA_C_real', 'Händelse, typ']]
    events_simu = events_simu[['rut_id', 'geometry', 'N_HA_C', 'Händelse, typ']]

    for ha in HA:
        events_real_ha = events_real.loc[events_real['Händelse, typ'] == ha]
        events_simu_ha = events_simu.loc[events_simu['Händelse, typ'] == ha]

        events_real_ha.to_file(data_folder / 'plots' / f'{ha}_real.gpkg', driver='GPKG')
        events_simu_ha.to_file(data_folder / 'plots' / f'{ha}_simu.gpkg', driver='GPKG')


def plot_events():
    data_folder = Path('data/output')
    events = pd.read_csv(data_folder / 'estimated_simulated_events.csv')
    output_path = data_folder / 'plots' / 'event_plots.pdf'

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


def run_simulation_single_event():
    data_folder = Path('data/output')

    area_C_MA_all = pd.read_pickle(data_folder / 'grid.pkl').set_index('rut_id').drop(columns='geometry')
    beta_MA_HA_all = pd.read_csv(data_folder / 'coefficients.csv', index_col=0)

    area_C_MA_all = area_C_MA_all.assign(intercept=1)
    area_C_MA_all = area_C_MA_all[[col for col in beta_MA_HA_all.columns]]

    # Create a separate cell for each MA that contains only that specific MA
    area_C_MA_all = pd.DataFrame(np.eye(len(area_C_MA_all.columns), dtype=int),
                                 columns=area_C_MA_all.columns,
                                 index=area_C_MA_all.columns).assign(intercept=1)

    assert list(area_C_MA_all.columns) == list(beta_MA_HA_all.columns), 'Datasets needs to have the same MA cols.'

    estimated_simulated_events = []

    # Test for a single event type TODO: Run and get results for additional event types
    beta_MA_HA_all = beta_MA_HA_all.loc[['Hjärtstopp']]

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

    pd.DataFrame(estimated_simulated_events).to_csv(data_folder / 'estimated_simulated_events_test.csv')


def run_simulation():
    data_folder = Path('data/output')

    area_C_MA_all = pd.read_pickle(data_folder / 'grid.pkl').set_index('rut_id').drop(columns='geometry')
    beta_MA_HA_all = pd.read_csv(data_folder / 'coefficients.csv', index_col=0)

    area_C_MA_all = area_C_MA_all / (1000 * 1000)
    area_C_MA_all = area_C_MA_all.assign(intercept=1)
    area_C_MA_all = area_C_MA_all[[col for col in beta_MA_HA_all.columns]]

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

    real_events = (pd.read_pickle(data_folder / 'processed_events.pkl')[['Ärende, årsnr', 'rut_id', 'Händelse, typ']]
                   .drop_duplicates(subset='Ärende, årsnr')
                   .drop(columns='Ärende, årsnr')
                   .groupby(['rut_id', 'Händelse, typ'])
                   .size()
                   .reset_index(name='N_HA_C_real'))

    events = events.merge(real_events, on=['rut_id', 'Händelse, typ'], how='outer').fillna(0)

    events.to_csv(data_folder / 'estimated_simulated_events.csv')


def main():
    run_simulation()


if __name__ == '__main__':
    main()
