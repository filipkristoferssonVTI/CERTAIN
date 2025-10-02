from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats
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


def clean_events_data(df):
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

    # TODO: Do we want to remove 'Resurs, återkallad' = Ja?
    # TODO: Should we only keep 'Resurs, uppgift' = INS?

    return df


def get_grid(data_folder, selected_county):
    mark = gpd.read_file(data_folder / 'gis' / 'Topografi 50 Nedladdning, vektor' / 'mark_sverige.gpkg', layer='mark')
    grid = gpd.read_file(data_folder / 'gis' / 'scb' / 'TotRut_SweRef.gpkg')
    counties = gpd.read_file(
        data_folder / 'gis' / 'Topografi 1M Nedladdning, vektor' / 'administrativindelning_sverige.gpkg',
        layer='lansyta')

    case_area = counties.loc[counties['lansnamn'] == selected_county]

    grid = gpd.sjoin(grid, case_area, how='inner', predicate='intersects')
    grid = grid[['rut_id', 'geometry']]

    intersections = gpd.overlay(grid, mark, how='intersection')
    intersections['objekttyp_area'] = intersections.geometry.area

    aggregated = intersections.groupby(['rut_id', 'objekttyp'], as_index=False).agg({'objekttyp_area': 'sum'})

    result = aggregated.pivot_table(index='rut_id', columns='objekttyp', values='objekttyp_area', fill_value=0)
    return grid.set_index('rut_id').join(result, how='left').reset_index()


def get_events(data_folder, grid):
    events = pd.read_csv(data_folder / 'Kopia av Daedalos export - Insatta resurser 2201 2411.csv',
                         sep=';',
                         skipfooter=1,
                         engine='python')

    events = clean_events_data(events)
    events = assign_geometry(events, x_col='Geo, ost', y_col='Geo, nord', crs='EPSG:3006')

    return gpd.sjoin(events, grid[['geometry', 'rut_id']], predicate="within")


def assign_geometry(df, x_col, y_col, crs):
    geometry = [Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else pd.NA for xy in zip(df[x_col], df[y_col])]
    return gpd.GeoDataFrame(df, geometry=geometry).set_crs(crs)


def print_confidence(y, y_pred, confidence):
    conf_intervals = np.array([stats.poisson.interval(confidence, m) for m in y_pred])

    within_band = (y >= conf_intervals[:, 0]) & (y <= conf_intervals[:, 1])
    oneormore = (y_pred >= 1)
    within_band_and_oneormore = (within_band & oneormore)
    print("    ", np.round(100 * np.sum(within_band) / y.shape[0], 2), "% within ", confidence, "-confidence")
    print("    ", np.round(np.sum(oneormore), 2), "observations >=1")
    print("    ", np.round(100 * np.sum(within_band_and_oneormore) / np.sum(oneormore), 2),
          "% of >=1 observations within confidence")


def poisson_neg_log_likelihood(beta, X, y):
    """Define the Poisson negative log-likelihood."""
    linear_pred = X @ beta
    expected_counts = np.exp(linear_pred)  # Poisson mean
    return -np.sum(y * linear_pred - expected_counts)  # negative log-likelihood


def generate_coefficients(events, grid):
    events = events.copy()
    grid = grid.copy()

    events = events.drop_duplicates(subset='Ärende, årsnr')  # to not count the same event more than once.

    design_matrix_cols = [col for col in grid.columns if col not in ['rut_id', 'geometry']]

    area_threshold = 1e6
    below_threshold = [col for col in design_matrix_cols if grid[col].sum() < area_threshold]
    print(f"Columns with sum less than {area_threshold}: {below_threshold}, these are dropped.")
    design_matrix_cols = [col for col in design_matrix_cols if col not in below_threshold]

    coeffs = {}

    for event_type in events['Händelse, typ'].dropna().unique():
        event_count = (events[['rut_id', 'Händelse, typ']]
                       .loc[events['Händelse, typ'] == event_type]
                       .groupby('rut_id')
                       .count()
                       .reset_index())

        g = grid.merge(event_count, how='left', on='rut_id').fillna(0)
        y = g['Händelse, typ'].values

        design_matrix = g[design_matrix_cols].values
        # An intercept columns is added with ones
        design_matrix = np.column_stack((design_matrix / 1e6, np.ones(design_matrix.shape[0])))

        n_samples, n_features = design_matrix.shape

        beta_init = np.zeros(n_features)
        result = opt.minimize(poisson_neg_log_likelihood, beta_init, args=(design_matrix, y), method="L-BFGS-B")
        coefficients = result.x

        print("... ", result.message)

        linear_pred = design_matrix @ coefficients
        y_pred = np.exp(linear_pred)  # Poisson mean
        coeffs[event_type] = coefficients

        print_confidence(y, y_pred, 0.95)

    design_matrix_cols.append('intercept')

    return pd.DataFrame(coeffs.values(),
                        columns=design_matrix_cols,
                        index=[event_type for event_type in coeffs.keys()])


def generate_lambda_vals(grid, coefficients):
    area_C_MA_all = grid.copy().set_index('rut_id').drop(columns='geometry')
    beta_MA_HA_all = coefficients.copy()

    area_C_MA_all = area_C_MA_all / (1000 * 1000)
    area_C_MA_all = area_C_MA_all.assign(intercept=1)
    area_C_MA_all = area_C_MA_all[[col for col in beta_MA_HA_all.columns]]

    assert list(area_C_MA_all.columns) == list(beta_MA_HA_all.columns), 'Datasets needs to have the same MA cols.'

    lambda_vals = []

    for C, area_C_MA in area_C_MA_all.iterrows():
        for HA, beta_MA_HA in beta_MA_HA_all.iterrows():
            lambda_HA_C = np.exp(area_C_MA.values @ beta_MA_HA.values)

            lambda_vals.append({
                'cell_id': C,
                'event_type': HA,
                'lambda': lambda_HA_C,  # estimated nr of events
            })

    return pd.DataFrame(lambda_vals)


def get_od_matrix(data_folder) -> pd.DataFrame:
    od_matrix = pd.read_csv(data_folder / 'od_matrix.csv', sep=',', skipfooter=1, engine='python',
                            dtype={'destination_id': str})
    od_matrix = od_matrix.rename(columns={'origin_id': 'fire_station',
                                          'destination_id': 'cell_id',
                                          'total_cost': 'dist_m'})
    od_matrix = od_matrix.drop(['entry_cost', 'network_cost', 'exit_cost'], axis=1)

    name_2_id = {
        'Centrum': '1000',
        'Kvillinge': '1200',
        'Skärblacka': '1500',
        'Östra Husby': '1600',
        'Krokek': '1900',
        'Kallerstad': '2200',
        'Lambohov': '2000',
        'Ulrika': '2500',
        'Ljungsbro': '2700',
        'Vikingstad': '2800',
        'Bestorp': '2900',
        'Söderköping': '3500',
        'Östra Ryd': '3600',
        'Bottna': '3700',
        'Valdemarsvik': '7000',
        'Åtvidaberg': '7500',
    }

    od_matrix['station_id'] = od_matrix['fire_station'].map(name_2_id)
    print(f'Dropping stations: {od_matrix.loc[od_matrix["station_id"].isna(), "fire_station"].unique().tolist()}')
    od_matrix = od_matrix.dropna(subset='station_id')
    od_matrix = od_matrix.dropna(subset='dist_m')  # TODO: How do we want to handle cells not reachable?
    return od_matrix


def get_energy_table(data_folder):
    energy_table = pd.read_excel(data_folder / 'Energianvändning_fordonskoder.xlsx')
    energy_table.index = energy_table['Fordonskod'].str[-2:]

    veh_types = ['10', '80', '60', '70', '40', '30', '65', '08', '67', '20']

    # TODO: We need to handle/add the missing vehicle types to the energy table.
    #  Are some types interchangeable and should be treated as one type therefore?
    missing_veh_types = [v for v in veh_types if v not in energy_table.index]
    print(f'Vehicle types not included in energy_table: {missing_veh_types}, these are treated as vehicle type 80.')
    new_rows = pd.DataFrame([energy_table.loc['80']] * len(missing_veh_types), index=missing_veh_types)
    energy_table = pd.concat([energy_table, new_rows])

    energy_table['battery_cap'] = energy_table['Batterikapacitet (kWh, motor)']
    energy_table['energy_need'] = energy_table['Energibehov (kWh/km)'] / 1000
    energy_table['charge_time'] = energy_table['Laddningstid (h, motor)'] * 3600
    energy_table['max_speed'] = 33  # m/s

    energy_table = energy_table.reset_index(names='veh_type')

    return energy_table[[
        'veh_type'
        'energy_need',
        'battery_cap',
        'charge_time',
        'max_speed',
    ]]


def main():
    data_folder = Path('../data')
    output_folder = data_folder / 'output'

    od = get_od_matrix(data_folder)
    grid = get_grid(data_folder, 'Östergötlands län')
    events = get_events(data_folder, grid)
    coefficients = generate_coefficients(events, grid)
    lambda_vals = generate_lambda_vals(grid, coefficients)
    energy_table = get_energy_table(data_folder)

    event_cols_2_keep = ['Ärende, årsnr', 'Resurs, station', 'Resurs, enhet', 'veh_type', 'Händelse, typ',
                         'Resurs, tid framme', 'Resurs, tid klar']
    events = events.drop(columns=[c for c in events.columns if c not in event_cols_2_keep])

    coefficients.to_csv(output_folder / 'coefficients.csv')

    energy_table.to_json(output_folder / 'energy_table.json', force_ascii=False, indent=2, orient='records')
    od.to_json(output_folder / 'od_matrix.json', force_ascii=False, indent=2, orient='records')
    lambda_vals.to_json(output_folder / 'lambda_vals.json', force_ascii=False, indent=2, orient='records')
    grid.drop(columns="geometry").to_json(output_folder / 'grid.json', force_ascii=False, indent=2, orient='records')
    events.to_json(output_folder / 'processed_events.json', force_ascii=False, indent=2, orient='records',
                   date_format="iso")


if __name__ == '__main__':
    main()
