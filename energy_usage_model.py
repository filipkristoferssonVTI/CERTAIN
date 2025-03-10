#%%
from pathlib import Path

import contextily as cx
import geopandas as gpd
import pandas as pd
import plotly.express as px
from shapely import Point

from gis_analysis import create_population_grid

battery_capacity = 540 #kWh

def calculate_efficiency(v):
    # https://publications.lib.chalmers.se/records/fulltext/133658.pdf
    # Formula and rolling resitance value from page 26
    # Other values from page 31, Volvo FH Summer
    #
    # Literature says the FH electric takes 1.1 kWh/km, numbers below gives 1.04 kWh/km.
    # Can probably be calibrated
    #
    # Assumes v in km/h, conversion to m/s in function

    v = v/3.6 # m/s, vehicle velocity

    rho = 1.228 # km/m^3, air density
    Cd = 0.59 # const, drag coefficient
    A = 9.7 # m^2, frontal area
    m = 40000 # kg, mass of vehicle
    g = 9.82 # m/s^2, gravitational acceleration (Sweden)
    fr = 0.0052 # rolling resistance 

    conv_factor = 2.77/10000 # Converting from N to kWh/km
    
    return (1/2*rho*Cd*A*v*v + m*g*fr)*conv_factor


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

    # Process fire station data into gdf
    fire_stations = process_fire_stations(data_folder)

    # Make grid gdf that corresponds with Filips grid
    grid = create_population_grid(data_folder)
    grid['rut_id'].astype(str)

    # Process OD-matrix from GIS and add to grid gdf
    od_matrix = process_od_matrix(data_folder)
    grid = pd.merge(grid, od_matrix, on='rut_id', how='inner')
    grid.to_file(data_folder / 'output' / 'grid_od_matrix.gpkg', driver='GPKG')

    # Plot firestations on a map
    plot_fire_stations_map(fire_stations)

    # Plot choropleth map of distance to grid cell from firestation
    plot_distance_from_firestation_map(grid, fire_stations, fire_station_plot)

    velocity_list = list(range(5,150,5))
    efficiency_list = [calculate_efficiency(v) for v in velocity_list]

    fig = px.scatter(x = velocity_list, y = efficiency_list)
    fig.show()

    grid['energy_usage'] = calculate_kWh_travel(grid['dist_km'])

    



if __name__ == '__main__':
    main()
# %%
