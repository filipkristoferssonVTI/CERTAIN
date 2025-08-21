from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_handelse_population(handelse_df, grid_df):
    df = grid_df.merge(handelse_df, on='rut_id', how='left')

    grouped = df.groupby(['handelse', 'rut_id', 'POP']).size().reset_index(name='occurrences')

    unique_handelse = grouped['handelse'].unique()
    num_objects = len(unique_handelse)
    fig = make_subplots(
        rows=num_objects,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2 / num_objects,
        subplot_titles=[f'{obj}' for obj in unique_handelse]
    )

    for i, handelse in enumerate(unique_handelse):
        obj_df = grouped[grouped['handelse'] == handelse]
        scatter = go.Scatter(x=obj_df['POP'], y=obj_df['occurrences'], mode='markers', name=handelse)
        fig.add_trace(scatter, row=i + 1, col=1)

    fig.update_layout(
        height=600 * num_objects,
        title_text="Occurrences Per Cell by Population (Shared X-Axis)",
        showlegend=False,
        template="plotly_white",
    )

    fig.update_xaxes(title_text="Population", row=num_objects, col=1)

    return fig


def plot_handelse_mark(handelse_df, grid_df):
    grid_df = grid_df.drop(columns='POP').set_index('rut_id')

    unique_handelse = handelse_df['handelse'].dropna().unique()

    mark_cols = [col for col in grid_df.columns if col != 'POP']
    grid_df['tot_area'] = grid_df[mark_cols].sum(axis=1)

    figs = []

    for i, handelse in enumerate(unique_handelse):
        h = handelse_df.loc[handelse_df['handelse'] == handelse].groupby('rut_id').count()
        grid_df_copy = grid_df.copy()
        grid_df_copy = grid_df_copy.join(h).fillna(0)

        numerator_lambda_hm = (grid_df_copy[mark_cols]
                               .div(grid_df_copy['tot_area'], axis=0)
                               .mul(grid_df_copy['handelse'], axis=0)
                               .sum(axis=0))
        denominator_lambda_hm = grid_df_copy[mark_cols].sum(axis=0)
        lambda_hm = numerator_lambda_hm / denominator_lambda_hm

        grid_df_copy['handelse_predicted'] = (grid_df_copy[mark_cols] * lambda_hm).sum(axis=1)

        scatter = px.scatter(
            x=grid_df_copy['handelse'],
            y=grid_df_copy['handelse_predicted'],
            title=handelse,
            trendline='ols'
        )
        figs.append(scatter)
    return figs


def main():
    data_folder = Path('data/output/eda')
    handelse = pd.read_csv(data_folder / 'ärenden.csv')
    grid = pd.read_csv(data_folder / 'grid_mark.csv')

    # fig1 = plot_handelse_population(handelse, grid)
    # fig1.write_html(data_folder / "handelse_population.html")

    fig2 = plot_handelse_mark(handelse, grid)
    for fig in fig2:
        fig.write_html(data_folder / "handelse_mark" / f"{fig.layout.title.text}.html")

    # merged_df = grid.merge(handelse, on='rut_id', how='left')
    # merged_df = pd.get_dummies(merged_df, columns=['handelse'], prefix='', prefix_sep='')
    # profile = ProfileReport(merged_df, title="EDA Report", explorative=True)
    # profile.to_file(data_folder / "eda_report.html")


if __name__ == '__main__':
    main()
