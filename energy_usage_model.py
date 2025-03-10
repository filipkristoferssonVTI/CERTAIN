#%%
from pathlib import Path

import contextily as cx
import geopandas as gpd
import pandas as pd
import plotly.express as px
from shapely import Point

from gis_analysis import create_population_grid

battery_capacity = 540 #kWh


def calculate_kWh_travel(km):
    # This energy efficiency is based on a 40 ton volvo electric truck, with an average speed of 80 km/h
    # https://www.volvotrucks.com/en-en/news-stories/press-releases/2022/jan/volvos-heavy-duty-electric-truck-is-put-to-the-test-excels-in-both-range-and-energy-efficiency.html
    eta = 1.1 # energy efficiency kWh/km
    return eta * km


def process_fire_stations(data_folder):
    fire_stations = gpd.read_file(
        data_folder / 'gis' / 'brandstationer' / 'brandstationer.gpkg')
    
    return fire_stations

def process_od_matrix(data_folder):
    od_matrix = pd.read_csv(data_folder / 'od_matrix' /'od_matrix.csv', sep=',', skipfooter=1, engine='python')

    od_matrix = od_matrix.rename(columns={'origin_id': 'fire_station', 
                                          'destination_id': 'rut_id',
                                          'total_cost': 'dist_m'})
    
    od_matrix['dist_km'] = od_matrix['dist_m']/1000
    od_matrix['rut_id'] = od_matrix['rut_id'].astype(str)
    
    od_matrix = od_matrix.drop(['entry_cost', 'network_cost', 'exit_cost'], axis = 1)

    return od_matrix

def plot_fire_stations_map(firestation_df):
    fig = firestation_df.plot(marker='o', color='red', markersize=15)

    fig.set_axis_off()

    cx.add_basemap(fig, crs=firestation_df.crs)

def plot_distance_from_firestation_map(dist_df, firestation_df, fire_station):

    dist_df = dist_df[(dist_df['fire_station'] == fire_station)].reset_index()
    firestation_df = firestation_df[(firestation_df['Namn'] == fire_station)].reset_index()

    fig = dist_df.plot(column='dist_km',
                       legend=True,
                       cmap='OrRd',
                       alpha=0.5,
                       missing_kwds={'color': 'darkgrey'})
    
    fig.set_axis_off()

    # Check crs
    firestation_df = firestation_df.to_crs(dist_df.crs)

    fig = firestation_df.plot(ax=fig, marker='o', color='red', markersize=15)

    cx.add_basemap(fig, crs=dist_df.crs)

def main():
    fire_station_plot = 'Vikingstad'

    data_folder = Path('data')

    fire_stations = process_fire_stations(data_folder)

    grid = create_population_grid(data_folder)
    grid['rut_id'].astype(str)

    od_matrix = process_od_matrix(data_folder)

    grid = pd.merge(grid, od_matrix, on='rut_id', how='inner')
    grid.to_file(data_folder / 'output' / 'grid_od_matrix.gpkg', driver='GPKG')

    plot_fire_stations_map(fire_stations)

    plot_distance_from_firestation_map(grid, fire_stations, fire_station_plot)

    grid['energy_usage'] = calculate_kWh_travel(grid['dist_km'])



if __name__ == '__main__':
    main()
# %%
