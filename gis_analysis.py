from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
from shapely import Point

from data_analysis import clean_data, group_data


def assign_geometry(df, x_col, y_col, crs):
    geometry = [Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else pd.NA for xy in zip(df[x_col], df[y_col])]
    return gpd.GeoDataFrame(df, geometry=geometry).set_crs(crs)


def count_points_in_polygons(polygons_gdf, points_gdf, polygon_id_col):
    points_in_polygons = gpd.sjoin(points_gdf, polygons_gdf, how='inner', predicate='within')
    point_counts = points_in_polygons.groupby(polygon_id_col).size()
    polygons_gdf['point_count'] = polygons_gdf[polygon_id_col].map(point_counts).fillna(0).astype(int)
    return polygons_gdf


def plot_choropleth(data):
    data = data.to_crs('EPSG:4326')
    fig = px.choropleth(data,
                        geojson=data.geometry,
                        locations=data.index,
                        color="point_count",
                        projection='mercator')
    fig.update_geos(fitbounds="locations", visible=False)
    fig.show()


def create_population_grid(data_folder):
    counties = gpd.read_file(
        data_folder / 'gis' / 'Topografi 1M Nedladdning, vektor' / 'administrativindelning_sverige.gpkg',
        layer='lansyta')

    case_area = counties.loc[counties['lansnamn'] == 'Östergötlands län']

    grid = gpd.read_file(data_folder / 'gis' / 'scb' / 'TotRut_SweRef.gpkg')
    popu = gpd.read_file(data_folder / 'gis' / 'scb' / 'totalbefolkning_1km_231231.gpkg')

    grid = gpd.sjoin(grid, case_area, how='inner', predicate='intersects')

    grid = grid.merge(popu[['Ruta', 'POP']], how='left', left_on='rut_id', right_on='Ruta')
    grid['POP'] = grid['POP'].fillna(0).astype(int)
    return grid[['rut_id', 'POP', 'geometry']]


def main():
    data_folder = Path('data')

    df = pd.read_csv(data_folder / 'Kopia av Daedalos export - Insatta resurser 2201 2411.csv', sep=';', skipfooter=1,
                     engine='python')

    df = clean_data(df)
    df = group_data(df)

    gdf = assign_geometry(df, x_col='geo_ost', y_col='geo_nord', crs='EPSG:3006')

    grid = create_population_grid(data_folder)

    # TODO: DeSO vs 1x1km grid?
    grid = count_points_in_polygons(grid, gdf, 'rut_id')

    plot_choropleth(grid)

    # gdf.to_file(data_folder / 'output' / 'ärenden.gpkg', driver='GPKG')
    # deso.to_file(data_folder / 'output' / 'deso.gpkg', driver='GPKG')
    # grid.to_file(data_folder / 'output' / 'grid.gpkg', driver='GPKG')


if __name__ == '__main__':
    main()
