from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import Point


def handle_missing_values(group, target_cols):
    for col in target_cols:
        non_na_vals = group[col].dropna().unique()
        if len(non_na_vals) == 1:
            unique_val = non_na_vals[0]
            group[col] = group[col].fillna(unique_val)
        elif len(non_na_vals) > 1:
            print(f"Warning: Group '{group.name}' has multiple values in column '{col}': {non_na_vals}")
    return group


def add_time_of_year(df):
    df['dagtid'] = df['Tid, timme'].apply(lambda x: 1 if 7 <= x <= 18 else 0)
    df['vardag'] = df['Tid, veckodag'].str[:2].astype(int).apply(lambda x: 1 if x <= 5 else 0)
    df['summer'] = df['Tid, månad'].str[:2].astype(int).apply(lambda x: 1 if 6 <= x <= 8 else 0)
    df['spring'] = df['Tid, månad'].str[:2].astype(int).apply(lambda x: 1 if 3 <= x <= 5 else 0)
    df['autumn'] = df['Tid, månad'].str[:2].astype(int).apply(lambda x: 1 if 9 <= x <= 11 else 0)
    return df


def clean_data(df):
    df = df.convert_dtypes()

    df['veh_type'] = df['Resurs, enhet'].str[-2:]

    df['Geo, nord'] = df['Geo, nord'].str.replace(' ', '').replace('0', pd.NA).astype('Int64')
    df['Geo, ost'] = df['Geo, ost'].str.replace(' ', '').replace('0', pd.NA).astype('Int64')

    # 'Resurs, tid klar' är när resursen är klar på skadeplatsen.
    # 'Resurs, tid avslut' är när resursen är helt klar med ärendet, är återställd och klar för nya larm.
    df['Resurs, tid klar'] = pd.to_datetime(df['Resurs, tid klar'], format='%Y-%m-%d %H:%M', errors='coerce')
    df['Resurs, tid avslut'] = pd.to_datetime(df['Resurs, tid avslut'], format='%Y-%m-%d %H:%M', errors='coerce')

    df['Resurs, tid larm'] = pd.to_datetime(df['Resurs, tid larm'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['Resurs, tid kvittens'] = pd.to_datetime(df['Resurs, tid kvittens'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['Resurs, tid framme'] = pd.to_datetime(df['Resurs, tid framme'], format='%Y-%m-%d %H:%M', errors='coerce')

    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)  # strip leading and trailing whitespaces

    # We only allow a single unique value in each target_col per group.
    df = df.groupby('Ärende, årsnr').apply(
        lambda group: handle_missing_values(group, [
            'Plats, miljö',
            'Händelse, typ',
            'Tid, månad',
            'Tid, veckodag',
            'Tid, timme',
            'Geo, nord',
            'Geo, ost']), include_groups=False).reset_index()

    df = add_time_of_year(df)

    # TODO: Remove 'Resurs, körtid (min)' = 0, 'Resurs, återkallad' = Ja? Only 'Resurs, uppgift' = INS?

    return df


def get_grid(data_folder):
    mark = gpd.read_file(data_folder / 'gis' / 'Topografi 50 Nedladdning, vektor' / 'mark_sverige.gpkg', layer='mark')
    grid = gpd.read_file(data_folder / 'gis' / 'scb' / 'TotRut_SweRef.gpkg')
    counties = gpd.read_file(
        data_folder / 'gis' / 'Topografi 1M Nedladdning, vektor' / 'administrativindelning_sverige.gpkg',
        layer='lansyta')

    case_area = counties.loc[counties['lansnamn'] == 'Östergötlands län']

    grid = gpd.sjoin(grid, case_area, how='inner', predicate='intersects')
    grid = grid[['rut_id', 'geometry']]

    intersections = gpd.overlay(grid, mark, how='intersection')
    intersections['objekttyp_area'] = intersections.geometry.area

    aggregated = intersections.groupby(['rut_id', 'objekttyp'], as_index=False).agg({'objekttyp_area': 'sum'})

    result = aggregated.pivot_table(index='rut_id', columns='objekttyp', values='objekttyp_area', fill_value=0)
    return grid.set_index('rut_id').join(result, how='left').reset_index()


def assign_geometry(df, x_col, y_col, crs):
    geometry = [Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else pd.NA for xy in zip(df[x_col], df[y_col])]
    return gpd.GeoDataFrame(df, geometry=geometry).set_crs(crs)


def main():
    data_folder = Path('data')

    grid = get_grid(data_folder)

    data = pd.read_csv(data_folder / 'Kopia av Daedalos export - Insatta resurser 2201 2411.csv',
                       sep=';',
                       skipfooter=1,
                       engine='python')

    data = clean_data(data)
    data = assign_geometry(data, x_col='Geo, ost', y_col='Geo, nord', crs='EPSG:3006')

    data = gpd.sjoin(data, grid[['geometry', 'rut_id']], predicate="within")

    grid.to_pickle(data_folder / 'output' / 'grid.pkl')
    data.to_pickle(data_folder / 'output' / 'processed_events.pkl')


if __name__ == '__main__':
    main()
