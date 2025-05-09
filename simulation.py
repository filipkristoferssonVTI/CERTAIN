import math
import random
from abc import abstractmethod, ABC
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from scipy.stats import poisson


class ResponseUnit(ABC):
    @abstractmethod
    def get_veh_types(self) -> list[tuple[str, int]]:
        """Returns a list of (veh_type, dur) tuples."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(veh_types='{self.get_veh_types()}')"


class ResponseUnitImpl(ResponseUnit):

    def __init__(self, vehicles: list[dict]):
        self._vehicles = vehicles

    def get_veh_types(self):
        max_dur = self.get_max_dur()
        return [(veh['veh_type'], max_dur) for veh in self._vehicles]

    def get_max_dur(self):
        return max([veh['dur'] for veh in self._vehicles])


@dataclass
class Vehicle:
    id: str
    station: str
    type: str
    battery_level: float
    next_avail_time: int = 0


class EventType(ABC):

    @property
    @abstractmethod
    def type(self) -> str:
        pass

    @abstractmethod
    def sample_response_unit(self) -> ResponseUnit:
        pass

    @abstractmethod
    def sample_start_time(self, T_start: int, T_end: int) -> int:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(type='{self.type}')"


class EventTypeImpl(EventType):

    def __init__(self, event_type: str, response_units: list[ResponseUnit]):
        self.event_type = event_type
        self.response_units = response_units

    @property
    def type(self):
        return self.event_type

    def sample_response_unit(self):
        return random.choice(self.response_units)

    def sample_start_time(self, T_start, T_end):
        return random.randint(T_start, T_end)


class Mission:

    def __init__(self, event_type: EventType, cell_id: str):
        self.event_type = event_type
        self.cell_id = cell_id
        self._response_unit = None
        self._start_time = None
        self._opt_response_time = None
        self._simu_response_time = None

    def __repr__(self):
        return (f"Mission(event_type={self.event_type!r}, "
                f"cell_id={self.cell_id!r}, "
                f"response_unit={self._response_unit!r}, "
                f"start_time={self._start_time!r}, "
                f"opt_response_time={self._opt_response_time!r}, "
                f"simu_response_time={self._simu_response_time!r})")

    def set_response_unit(self, response_unit: ResponseUnit):
        if self._response_unit is not None:
            raise ValueError("Response unit is already set!")
        self._response_unit = response_unit

    @property
    def response_unit(self):
        return self._response_unit

    def set_start_time(self, start_time: int):
        if self._start_time is not None:
            raise ValueError("Start time is already set!")
        self._start_time = start_time

    @property
    def start_time(self):
        return self._start_time

    def set_opt_response_time(self, time: int):
        if self._opt_response_time is not None:
            raise ValueError("Optimal response time is already set!")
        self._opt_response_time = time

    @property
    def opt_response_time(self):
        return self._opt_response_time

    def set_simu_response_time(self, time: int):
        if self._simu_response_time is not None:
            raise ValueError("Simulated response time is already set!")
        self._simu_response_time = time

    @property
    def simu_response_time(self):
        return self._simu_response_time

    def to_dict(self):
        return {'event_type': self.event_type,
                'cell_id': self.cell_id,
                'response_unit': self.response_unit,
                'start_time': self.start_time,
                'opt_response_time': self.opt_response_time,
                'simu_response_time': self.simu_response_time}


class MissionContainer(ABC):

    @abstractmethod
    def add_response_unit(self):
        pass

    @abstractmethod
    def add_start_time(self, T_start: int, T_end: int):
        pass

    @abstractmethod
    def sort_by_start_time(self):
        pass

    @abstractmethod
    def save(self, output_file: Path):
        pass


class MissionContainerImpl(MissionContainer):
    def __init__(self, missions: list[Mission]):
        self.missions = missions

    def add_response_unit(self):
        for mission in self.missions:
            mission.set_response_unit(mission.event_type.sample_response_unit())

    def add_start_time(self, T_start, T_end):
        for mission in self.missions:
            mission.set_start_time(mission.event_type.sample_start_time(T_start, T_end))
        self.sort_by_start_time()

    def sort_by_start_time(self):
        self.missions.sort(key=lambda m: m.start_time)

    def save(self, output_file):
        pd.DataFrame([m.to_dict() for m in self.missions]).to_csv(output_file, index=False)


class EnergyTable(ABC):
    # TODO: Refactor (rename) as this also holds info on vehicle max speed
    @abstractmethod
    def get_battery_cap(self, veh_type: str) -> float:
        pass

    @abstractmethod
    def get_energy_need(self, veh_type: str) -> float:
        pass

    @abstractmethod
    def get_charge_time(self, veh_type: str) -> int:
        pass

    @abstractmethod
    def get_max_speed(self, veh_type: str) -> float:
        pass


class EnergyTableImpl(EnergyTable):

    def __init__(self, energy_table: pd.DataFrame):
        self.energy_table = energy_table

    def get_battery_cap(self, veh_type):
        return self.energy_table.loc[veh_type, 'battery_cap']

    def get_energy_need(self, veh_type):
        return self.energy_table.loc[veh_type, 'energy_need']

    def get_charge_time(self, veh_type):
        return self.energy_table.loc[veh_type, 'charge_time']

    def get_max_speed(self, veh_type):
        return self.energy_table.loc[veh_type, 'max_speed']


class TravelTimeModel(ABC):

    @abstractmethod
    def get_sorted_stations(self, cell_id: str) -> list[str]:
        """Returns all stations sorted by the distance to the cell."""
        pass

    @abstractmethod
    def get_distance(self, station_id: str, cell_id: str) -> float:
        pass

    @abstractmethod
    def get_travel_time(self, max_speed: float, station_id: str, cell_id: str) -> int:
        pass


class TravelTimeModelImpl(TravelTimeModel):

    def __init__(self, od_matrix: pd.DataFrame):
        self.od = self._sort_od_matrix(od_matrix)

    @staticmethod
    def _sort_od_matrix(od_matrix):
        return od_matrix.sort_values(by=['cell_id', 'dist_m']).set_index('cell_id')

    def get_sorted_stations(self, cell_id):
        return self.od.loc[cell_id, 'station_id'].tolist()

    @lru_cache(maxsize=None)
    def get_distance(self, station_id, cell_id):
        distance = self.od.loc[(self.od.index == cell_id) & (self.od['station_id'] == station_id), 'dist_m']
        assert len(
            distance) == 1, f'A unique distance for the given station/cell combination needs to be found.'
        return distance.squeeze()

    def get_travel_time(self, max_speed, station_id, cell_id):
        distance = self.get_distance(station_id, cell_id)
        return math.ceil(distance / max_speed)


class VehicleContainer(ABC):

    @abstractmethod
    def simulate_mission(self, mission: Mission, travel_time_model: TravelTimeModel, energy_table: EnergyTable):
        pass

    @abstractmethod
    def simulate_movement(self, mission: Mission, vehicle: Vehicle, energy_table: EnergyTable,
                          travel_time_model: TravelTimeModel, dur: int) -> int:
        pass


class VehicleContainerImpl(VehicleContainer):

    def __init__(self, vehicles: list[Vehicle]):
        self.vehicles = self._structure_vehicles(vehicles)

    @staticmethod
    def _structure_vehicles(vehicles: list[Vehicle]) -> dict:
        vehicles_structured = defaultdict(list)
        for v in vehicles:
            vehicles_structured[v.type].append(v)
        return vehicles_structured

    def get_vehicle(self, veh_type: str, cell_id: str, travel_time_model: TravelTimeModel, energy_table: EnergyTable):
        max_speed = energy_table.get_max_speed(veh_type)
        return min(self.vehicles[veh_type],
                   key=lambda veh: veh.next_avail_time + travel_time_model.get_travel_time(max_speed,
                                                                                           veh.station,
                                                                                           cell_id))

    def get_opt_arrival_time(self, veh_type: str, mission: Mission, travel_time_model: TravelTimeModel,
                             energy_table: EnergyTable):
        max_speed = energy_table.get_max_speed(veh_type)
        cell_id = mission.cell_id

        vehicle = min(self.vehicles[veh_type], key=lambda veh: travel_time_model.get_travel_time(max_speed,
                                                                                                 veh.station,
                                                                                                 cell_id))
        tt = travel_time_model.get_travel_time(max_speed, vehicle.station, cell_id)
        return mission.start_time + tt

    def simulate_mission(self, mission, travel_time_model, energy_table):
        latest_arrival_opt = 0
        latest_arrival_simu = 0

        for (veh_type, dur) in mission.response_unit.get_veh_types():
            vehicle = self.get_vehicle(veh_type, mission.cell_id, travel_time_model, energy_table)

            arrival_time_opt = self.get_opt_arrival_time(veh_type, mission, travel_time_model, energy_table)
            arrival_time_simu = self.simulate_movement(mission, vehicle, energy_table, travel_time_model, dur)

            if arrival_time_simu > latest_arrival_simu:
                latest_arrival_simu = arrival_time_simu

            if arrival_time_opt > latest_arrival_opt:
                latest_arrival_opt = arrival_time_opt

        mission.set_opt_response_time(latest_arrival_opt)
        mission.set_simu_response_time(latest_arrival_simu)

    def simulate_movement(self, mission, vehicle, energy_table, travel_time_model, dur):
        # TODO: Include energy usage at site, use some default value?
        # TODO: Include charge time? Right now treated as "normal vehicles"?

        station_id = vehicle.station
        cell_id = mission.cell_id
        veh_type = vehicle.type

        tt = travel_time_model.get_travel_time(energy_table.get_max_speed(veh_type), station_id, cell_id)
        energy_usage = energy_table.get_energy_need(veh_type) * travel_time_model.get_distance(station_id, cell_id)

        start_time = mission.start_time
        if start_time < vehicle.next_avail_time:
            start_time = vehicle.next_avail_time

        arrival_time = start_time + tt

        vehicle.next_avail_time = arrival_time + dur + tt
        vehicle.battery_level = vehicle.battery_level - (energy_usage * 2)

        return arrival_time


class RegrModel(ABC):
    @abstractmethod
    def gen_missions(self, event_type: EventType) -> list[Mission]:
        """Generates a number of missions of a given event type."""
        pass


class RegrModelImpl(RegrModel):
    def __init__(self, lambda_vals: pd.DataFrame):
        self.lambda_vals = lambda_vals

    def gen_missions(self, event_type):
        filtered_lambda_vals = self.lambda_vals.loc[self.lambda_vals['event_type'] == event_type.type].copy()
        filtered_lambda_vals['occurrences'] = poisson.rvs(filtered_lambda_vals['lambda'])
        filtered_lambda_vals = filtered_lambda_vals.loc[filtered_lambda_vals['occurrences'] != 0]
        missions = []
        for _, row in filtered_lambda_vals.iterrows():
            missions.extend([
                Mission(event_type=event_type, cell_id=row['cell_id'])
                for _ in range(row['occurrences'])
            ])
        return missions


def create_response_units(real_events: pd.DataFrame, event_type: str) -> list[ResponseUnit]:
    def get_dur(arrived, finished):
        if pd.notna(arrived) and pd.notna(finished):
            return int((finished - arrived).total_seconds())
        elif mean_dur:
            return mean_dur
        else:
            raise ValueError(f'No duration found for event type: {event_type}.')

    # TODO: Get mean travel time for each vehicle type instead?
    mean_dur_df = real_events.dropna(subset=['Resurs, tid framme', 'Resurs, tid klar'])
    mean_dur = int((mean_dur_df['Resurs, tid klar'] - mean_dur_df['Resurs, tid framme']).dt.total_seconds().mean())

    event_type_group = real_events.loc[real_events['Händelse, typ'] == event_type].copy()
    response_units = []
    for _, event_group in event_type_group.groupby('Ärende, årsnr'):
        response_units.append(ResponseUnitImpl(
            vehicles=[{
                'veh_type': row['veh_type'],
                'dur': get_dur(row['Resurs, tid framme'], row['Resurs, tid klar'])
            } for _, row in event_group.iterrows()]))
    return response_units


def create_vehicles(real_events: pd.DataFrame, energy_table: EnergyTable) -> VehicleContainer:
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

        # TODO: One of these might be incorrect
        'Östra Ryd': '3600',
        'Bottna': '3900',

        'Valdemarsvik': '7000',
        'Åtvidaberg': '7500',
    }

    od_matrix['station_id'] = od_matrix['fire_station'].map(name_2_id)

    od_matrix = od_matrix.dropna(subset='station_id')  # TODO: Handle missing stations
    od_matrix = od_matrix.dropna(subset='dist_m')  # TODO: How do we want to handle cells not reachable?

    return od_matrix


def process_energy_table(energy_table: pd.DataFrame, veh_types: list[str]) -> pd.DataFrame:
    energy_table.index = energy_table['Fordonskod'].str[-2:]

    # TODO: How do we want to handle the missing vehicle types?
    missing_veh_types = [v for v in veh_types if v not in energy_table.index]
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
    path = Path('data')

    real_event_data = pd.read_pickle(path / 'output' / 'processed_events.pkl')
    lambda_vals = pd.read_csv(path / 'output' / 'lambda_vals.csv', dtype={'cell_id': str})
    od_matrix = process_od_matrix(
        pd.read_csv(path / 'od_matrix.csv', sep=',', skipfooter=1, engine='python', dtype={'destination_id': str}))
    energy_table = pd.read_excel(path / 'Energianvändning_fordonskoder.xlsx')

    # TODO: Remove once the stations are fixed
    real_event_data = real_event_data[
        real_event_data['Resurs, station'].str[-4:].isin(od_matrix['station_id'].unique())]

    energy_table = process_energy_table(energy_table, real_event_data['veh_type'].unique().tolist())

    energy_table = EnergyTableImpl(energy_table)
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
