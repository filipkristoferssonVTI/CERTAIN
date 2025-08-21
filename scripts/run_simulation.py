from pathlib import Path

import pandas as pd

from model.simulation.implementations import ResponseUnitImpl, EventTypeImpl, MissionContainerImpl, \
    VehicleDataTableImpl, TravelTimeModelImpl, VehicleContainerImpl, RegrModelImpl, Vehicle
from model.simulation.interfaces import ResponseUnit, VehicleDataTable, VehicleContainer


def create_response_units(real_events: pd.DataFrame, event_type: str) -> list[ResponseUnit]:
    """If dur info is missing for a vehicle, the mean dur of all vehicle within the same event type is used."""

    def get_dur(arrived, finished):
        if pd.notna(arrived) and pd.notna(finished):
            return int((finished - arrived).total_seconds())
        elif pd.notna(mean_dur):
            return mean_dur
        else:
            raise ValueError(f'No duration found for event type: {event_type}.')

    event_type_group = real_events.loc[real_events['Händelse, typ'] == event_type].copy()
    mean_dur_df = event_type_group.dropna(subset=['Resurs, tid framme', 'Resurs, tid klar'])
    mean_dur = (mean_dur_df['Resurs, tid klar'] - mean_dur_df['Resurs, tid framme']).dt.total_seconds().mean()

    response_units = []
    for _, event_group in event_type_group.groupby('Ärende, årsnr'):
        response_units.append(ResponseUnitImpl(
            vehicles=[{
                'veh_type': row['veh_type'],
                'dur': get_dur(row['Resurs, tid framme'], row['Resurs, tid klar'])
            } for _, row in event_group.iterrows()]))
    return response_units


def create_vehicles(real_events: pd.DataFrame, energy_table: VehicleDataTable) -> VehicleContainer:
    real_events = real_events.drop_duplicates(subset='Resurs, enhet').copy()
    vehicles = []
    for _, row in real_events.iterrows():
        vehicles.append(Vehicle(
            id=row['Resurs, enhet'][-4:],
            station=row['Resurs, station'][-4:],
            type=row['veh_type'],
            battery_level=energy_table.get_battery_cap(row['veh_type'])
        ))
    return VehicleContainerImpl(vehicles)


def process_od_matrix(od_matrix: pd.DataFrame) -> pd.DataFrame:
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


def process_energy_table(energy_table: pd.DataFrame, veh_types: list[str]) -> pd.DataFrame:
    energy_table.index = energy_table['Fordonskod'].str[-2:]

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

    return energy_table[[
        'energy_need',
        'battery_cap',
        'charge_time',
        'max_speed',
    ]]


def main():
    path = Path('../data')

    real_event_data = pd.read_pickle(path / 'output' / 'processed_events.pkl')
    lambda_vals = pd.read_csv(path / 'output' / 'lambda_vals.csv', dtype={'cell_id': str})
    od_matrix = process_od_matrix(
        pd.read_csv(path / 'od_matrix.csv', sep=',', skipfooter=1, engine='python', dtype={'destination_id': str}))
    energy_table = pd.read_excel(path / 'Energianvändning_fordonskoder.xlsx')

    # We only keep events that are associated with known stations
    real_event_data = real_event_data[
        real_event_data['Resurs, station'].str[-4:].isin(od_matrix['station_id'].unique())]

    known_veh_types = ['10', '80', '60', '70', '40', '30', '65', '08', '67', '20']

    # We only keep vehicles that are associated with known vehicle types
    real_event_data = real_event_data[real_event_data['veh_type'].isin(known_veh_types)]

    energy_table = process_energy_table(energy_table, real_event_data['veh_type'].unique().tolist())

    energy_table = VehicleDataTableImpl(energy_table)
    travel_time_model = TravelTimeModelImpl(od_matrix)
    regr_model = RegrModelImpl(lambda_vals)

    missions = []
    for event_type_str in lambda_vals['event_type'].unique():
        response_units = create_response_units(real_event_data, event_type_str)
        event_type = EventTypeImpl(event_type_str, response_units)
        missions.extend(regr_model.gen_missions(event_type))

    mission_container = MissionContainerImpl(missions)
    mission_container.add_response_unit()
    mission_container.add_start_time(T_start=0, T_end=3 * 365 * 24 * 60 * 60)

    vehicle_container = create_vehicles(real_event_data, energy_table)

    for mission in mission_container.missions:
        # TODO: Remove this once unreachable cells are handled
        if mission.cell_id in travel_time_model.od.index:
            vehicle_container.simulate_mission(mission, travel_time_model, energy_table)

    mission_container.save(path / 'output' / 'missions_test.csv')


if __name__ == '__main__':
    main()
