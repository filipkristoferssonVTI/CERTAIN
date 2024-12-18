from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import gaussian_kde


def save_df_to_json(df, filename, orient='records'):
    try:
        json_data = df.to_json(orient=orient, indent=4, force_ascii=False)
        with open(filename, 'w', encoding='utf-8') as json_file:
            json_file.write(json_data)

        print(f"DataFrame successfully saved to {filename} in JSON format.")
    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")


def check_column_uniformity(df, col):
    """Ignores NA."""
    non_nan_values = df[col].dropna()
    if non_nan_values.empty:
        return None
    unique_values = non_nan_values.unique()
    if len(unique_values) == 1:
        return unique_values[0]
    else:
        raise ValueError(f"The column '{col}' contains multiple unique non-NA values: {unique_values}")


def drop_invalid_rows(group):
    if group.duplicated(subset='Resurs, enhet', keep=False).any():
        group = group.loc[group.groupby('Resurs, enhet')['Resurs, tid total'].idxmax()]
    return group


def group_data(df):
    groups = df.groupby('Ärende, årsnr')

    data = []

    for name, group in groups:
        plats = check_column_uniformity(group, 'Plats, miljö')
        handelse = check_column_uniformity(group, 'Händelse, typ')
        manad = check_column_uniformity(group, 'Tid, månad')
        timme = check_column_uniformity(group, 'Tid, timme')
        geo_nord = check_column_uniformity(group, 'Geo, nord')
        geo_ost = check_column_uniformity(group, 'Geo, ost')

        # TODO: Only include certain 'Resurs, uppgift'?

        group = drop_invalid_rows(group)

        insatsstyrka = frozenset(group['veh_type'].value_counts(dropna=False).items())

        data.append({'plats': plats,
                     'handelse': handelse,
                     'manad': manad,
                     'timme': timme,
                     'geo_nord': geo_nord,
                     'geo_ost': geo_ost,
                     'insatsstyrka': insatsstyrka})

    return pd.DataFrame(data)


def clean_data(df):
    df = df.convert_dtypes()

    df['veh_type'] = df['Resurs, enhet'].str[-2:]

    df['Geo, nord'] = df['Geo, nord'].str.replace(' ', '').replace('0', pd.NA).astype('Int64')
    df['Geo, ost'] = df['Geo, ost'].str.replace(' ', '').replace('0', pd.NA).astype('Int64')

    df['Resurs, tid larm'] = pd.to_datetime(df['Resurs, tid larm'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['Resurs, tid kvittens'] = pd.to_datetime(df['Resurs, tid kvittens'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['Resurs, tid klar'] = pd.to_datetime(df['Resurs, tid klar'], format='%Y-%m-%d %H:%M', errors='coerce')
    df['Resurs, tid framme'] = pd.to_datetime(df['Resurs, tid framme'], format='%Y-%m-%d %H:%M', errors='coerce')
    df['Resurs, tid avslut'] = pd.to_datetime(df['Resurs, tid avslut'], format='%Y-%m-%d %H:%M', errors='coerce')

    # TODO: From Michael: "'Resurs, tid klar' är när resursen är klar på skadeplatsen,
    #  'Resurs, tid avslut' är när resursen är helt klar med ärendet, är återställd och klar för nya larm."
    #  How do we find the travel time back to the station (does the vehicle travel back)?

    df['Resurs, tid total'] = df['Resurs, tid klar'] - df['Resurs, tid larm']
    # To work in drop_invalid_rows
    df['Resurs, tid total'] = df['Resurs, tid total'].fillna(df['Resurs, tid total'].min())

    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)  # strip leading and trailing whitespaces
    return df


def plot_kde(data):
    kde = gaussian_kde(data)
    x_values = np.linspace(min(data), max(data), 1000)
    density = kde(x_values)

    plt.hist(data, bins='auto')
    plt.xlim(0, 60)
    plt.show()

    # KDE (Smooth Distribution)
    plt.plot(x_values, density)
    plt.show()


def plot_insatsstyrka_hist(data):
    insatsstyrka = data['insatsstyrka'].value_counts().reset_index()
    insatsstyrka['percentage'] = (insatsstyrka['count'] / insatsstyrka['count'].sum()) * 100
    insatsstyrka['cumulative_percentage'] = insatsstyrka['percentage'].cumsum()

    insatsstyrka['insatsstyrka'] = insatsstyrka['insatsstyrka'].apply(lambda x: ', '.join(f"{k}: {v}" for k, v in x))

    fig = px.bar(insatsstyrka, x='insatsstyrka', y='count')
    fig2 = px.bar(insatsstyrka, x='insatsstyrka', y='cumulative_percentage')
    fig.show()
    fig2.show()


def plot_handelse_insatsstyrka_coverage(data):
    insatsstyrka = data['insatsstyrka'].value_counts().reset_index()
    insatsstyrka['percentage'] = (insatsstyrka['count'] / insatsstyrka['count'].sum()) * 100
    insatsstyrka['cumulative_percentage'] = insatsstyrka['percentage'].cumsum()

    above_limit = insatsstyrka.loc[insatsstyrka['cumulative_percentage'] <= 95, 'insatsstyrka']
    data['above_limit'] = data['insatsstyrka'].isin(above_limit)

    coverage_data = (data.groupby('handelse')['above_limit']
                     .mean()  # Calculate the mean (percentage of True values)
                     .mul(100)  # Convert to percentage
                     .reset_index()
                     .rename(columns={'above_limit': 'coverage_percentage'})
                     .sort_values(by='coverage_percentage', ascending=False))

    fig = px.bar(coverage_data, y='coverage_percentage', x='handelse')
    fig.show()


def main():
    data_folder = Path('data')
    df = pd.read_csv(data_folder / 'Kopia av Daedalos export - Insatta resurser 2201 2411.csv', sep=';', skipfooter=1,
                     engine='python')

    df = clean_data(df)
    df = group_data(df)

    plot_handelse_insatsstyrka_coverage(df)


if __name__ == '__main__':
    main()
