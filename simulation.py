import random
from abc import abstractmethod, ABC
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import poisson


class ResponseUnit(ABC):
    @abstractmethod
    def get_veh_types(self) -> list[str]:
        pass

    @abstractmethod
    def get_dur(self) -> float:
        pass


class ResponseUnitImpl(ResponseUnit):

    def __init__(self, vehicles: list[dict]):
        self._vehicles = vehicles

    def get_veh_types(self):
        return [veh['veh_type'] for veh in self._vehicles]

    def get_dur(self):
        # TODO: Handle NA?
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


class TravelTimeModel(ABC):
    @abstractmethod
    def get_travel_time(self, mission: Mission) -> int:
        pass

    @abstractmethod
    def get_closest_station(self, mission: Mission) -> str:
        pass


class ODMatrix(ABC):

    @abstractmethod
    def get_closest_station_id(self, cell_id: str) -> str:
        pass

    @abstractmethod
    def get_distance(self, station_id: str, cell_id: str) -> float:
        pass


class ODMatrixImpl(ODMatrix):
    def __init__(self, od_matrix: pd.DataFrame):
        self.od = od_matrix
        self.closest_stations = self._create_closest_stations()

    def _create_closest_stations(self):
        return self.od.loc[self.od.groupby('cell_id')['dist_m'].idxmin()].set_index('cell_id')

    def get_closest_station_id(self, cell_id):
        return self.closest_stations.loc[cell_id, 'station_id']

    def get_distance(self, station_id, cell_id):
        dist_m = self.od.loc[(self.od['station_id'] == station_id) & (self.od['cell_id'] == cell_id), 'dist_m']
        assert len(dist_m) == 1, 'A unique distance for the given station/cell combination needs to be found.'
        return dist_m.iloc[0]


class EnergyTable(ABC):
    @abstractmethod
    def get_battery_cap(self, veh_type: str) -> float:
        pass


class EnergyTableImpl(EnergyTable):

    def __init__(self, energy_table: pd.DataFrame):
        self.energy_table = energy_table

    def get_battery_cap(self, veh_type: str) -> float:
        if veh_type not in self.energy_table.index:
            return 300  # TODO: Handle missing vehicle types
        return self.energy_table.loc[veh_type, 'Batterikapacitet (kWh, motor)']


class TravelTimeModelImpl(TravelTimeModel):

    def __init__(self, od_matrix: ODMatrix):
        self.od = od_matrix

    def get_travel_time(self, mission):
        pass

    def get_closest_station(self, mission):
        return self.od.get_closest_station_id(mission.cell_id)


class VehicleContainer(ABC):

    @abstractmethod
    def get_avail_vehicles(self, mission: Mission, travel_time_model: TravelTimeModel) -> list[Vehicle]:
        pass


class VehicleContainerImpl(VehicleContainer):

    def __init__(self, vehicles: list[Vehicle]):
        self.vehicles = self.structure_vehicles(vehicles)

    @staticmethod
    def structure_vehicles(vehicles: list[Vehicle]) -> dict:
        vehicles_structured = defaultdict(list)
        for v in vehicles:
            vehicles_structured[(v.station, v.type)].append(v)
        return vehicles_structured

    def get_avail_vehicles(self, mission, travel_time_model):
        # TODO: If no vehicle of a type is avail?
        veh_types = mission.response_unit.get_veh_types()
        closest_station = travel_time_model.get_closest_station(mission)
        avail_vehicles = []
        for veh_type in veh_types:
            vehicles = self.vehicles[(closest_station, veh_type)]
            for vehicle in vehicles:
                if vehicle.next_avail_time < mission.start_time:
                    avail_vehicles.append(vehicle)
                    break
        return avail_vehicles


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
    event_type_group = real_events.loc[real_events['Händelse, typ'] == event_type].copy()
    response_units = []
    for _, event_group in event_type_group.groupby('Ärende, årsnr'):
        response_units.append(ResponseUnitImpl(
            vehicles=[{'veh_type': row['veh_type'],
                       'dur': row['Resurs, tid klar'] - row['Resurs, tid framme']}
                      for _, row in event_group.iterrows()]))
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


def process_od_matrix(data_path):
    od_matrix = pd.read_csv(data_path, sep=',', skipfooter=1, engine='python', dtype={'destination_id': str})
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
        'Bottna': '3600',
        'Valdemarsvik': '7000',
        'Åtvidaberg': '7500',
    }

    od_matrix['station_id'] = od_matrix['fire_station'].map(name_2_id)

    od_matrix = od_matrix.dropna(subset='station_id')  # TODO: Handle missing stations
    od_matrix = od_matrix.dropna(subset='dist_m')  # TODO: How do we handle cells not reachable?

    return od_matrix


def main():
    data_folder = Path('data')

    real_event_data = pd.read_pickle(data_folder / 'output' / 'processed_events.pkl')
    lambda_vals = pd.read_csv(data_folder / 'output' / 'lambda_vals.csv', dtype={'cell_id': str})
    od_matrix = process_od_matrix(data_folder / 'od_matrix.csv')
    energy_table = pd.read_excel(data_folder / 'Energianvändning_fordonskoder.xlsx')
    energy_table.index = energy_table['Fordonskod'].str[-2:]

    energy_table = EnergyTableImpl(energy_table)

    travel_time_model = TravelTimeModelImpl(ODMatrixImpl(od_matrix))

    event_type_str = 'Hjärtstopp'

    vehicle_container = create_vehicles(real_event_data, energy_table)
    response_units = create_response_units(real_event_data, event_type_str)

    regr_model = RegrModelImpl(lambda_vals)
    event_type = EventTypeImpl(event_type_str, response_units)

    mission_container = MissionContainerImpl(regr_model.gen_missions(event_type))
    mission_container.add_response_unit()
    mission_container.add_start_time(T_start=0, T_end=3 * 365 * 24 * 60 * 60)

    for mission in mission_container.missions:
        avail_vehicles = vehicle_container.get_avail_vehicles(mission, travel_time_model)
        DEBUG = 1


if __name__ == '__main__':
    main()
