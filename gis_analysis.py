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
    grid = gpd.read_file(data_folder / 'gis' / 'scb' / 'TotRut_SweRef.gpkg')
    popu = gpd.read_file(data_folder / 'gis' / 'scb' / 'totalbefolkning_1km_231231.gpkg')

    grid = grid.merge(popu[['Ruta', 'POP']], how='left', left_on='rut_id', right_on='Ruta')
    grid['POP'] = grid['POP'].fillna(0).astype(int)
    return grid[['rut_id', 'POP', 'geometry']]


def process_arenden(data_folder):
    df = pd.read_csv(data_folder / 'Kopia av Daedalos export - Insatta resurser 2201 2411.csv', sep=';', skipfooter=1,
                     engine='python')
    df = clean_data(df)
    df = group_data(df)
    gdf = assign_geometry(df, x_col='geo_ost', y_col='geo_nord', crs='EPSG:3006')
    return gdf


def merge_grid_mark(grid, mark):
    intersections = gpd.overlay(grid, mark, how='intersection')
    intersections['objekttyp_area'] = intersections.geometry.area

    aggregated = intersections.groupby(['rut_id', 'objekttyp'], as_index=False).agg({'objekttyp_area': 'sum'})

    result = aggregated.pivot_table(index='rut_id', columns='objekttyp', values='objekttyp_area', fill_value=0)
    return grid.set_index('rut_id').join(result, how='left').reset_index()


def merge_grid_road(grid, road):
    intersections = gpd.sjoin(grid, road, how='left', predicate='intersects')

    presence = (
        intersections.groupby(['rut_id', 'Typ'])
        .size()
        .unstack(fill_value=0)
        .map(lambda x: 1 if x > 0 else 0)
    )

    return grid.set_index('rut_id').join(presence, how='left').fillna(0).reset_index()


def main():
    data_folder = Path('data')

    road = gpd.read_file(data_folder / 'gis' / 'lastkajen' / 'Vägslag_516890.gpkg')
    mark = gpd.read_file(data_folder / 'gis' / 'Topografi 50 Nedladdning, vektor' / 'mark_sverige.gpkg', layer='mark')
    grid = create_population_grid(data_folder)
    arenden = process_arenden(data_folder)

    plats_mark = gpd.sjoin(arenden, mark, how="inner", predicate="within")
    plats_mark_ct = pd.crosstab(plats_mark["plats"], plats_mark["objekttyp"])

    # We only need the cells that actually contains an arende in this section.
    grid = grid[grid.intersects(arenden.union_all())]

    grid = merge_grid_mark(grid, mark)
    grid = merge_grid_road(grid, road)

    # For EDA
    cols_2_modify = [col for col in grid.columns if col not in ['rut_id', 'POP', 'geometry']]
    grid[cols_2_modify] = grid[cols_2_modify].map(lambda x: True if x > 0 else False)
    arenden = gpd.sjoin(arenden, grid[['geometry', 'rut_id']], how="left", predicate="within")
    arenden = arenden[['rut_id', 'handelse']]
    grid = grid.drop(columns=['geometry'])
    arenden.to_csv(data_folder / 'output' / 'eda' / 'ärenden.csv', index=False)
    grid.to_csv(data_folder / 'output' / 'eda' / 'grid.csv', index=False)

    # grid = count_points_in_polygons(grid, arenden, 'rut_id')

    # plats_mark_ct.to_csv(data_folder / 'output' / 'plats_mark_ct.csv')
    # arenden.to_file(data_folder / 'output' / 'ärenden.gpkg', driver='GPKG')
    # grid.to_file(data_folder / 'output' / 'grid.gpkg', driver='GPKG')


if __name__ == '__main__':
    main()
