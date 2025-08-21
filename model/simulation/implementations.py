import math
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from scipy.stats import poisson

from model.simulation.interfaces import *


@dataclass
class Vehicle:
    id: str
    station: str
    type: str
    battery_level: float
    next_avail_time: int = 0


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
        if self._response_unit is None:
            raise ValueError("Response unit needs to be set!")
        return self._response_unit

    def set_start_time(self, start_time: int):
        if self._start_time is not None:
            raise ValueError("Start time is already set!")
        self._start_time = start_time

    @property
    def start_time(self):
        if self._start_time is None:
            raise ValueError("Start time needs to be set!")
        return self._start_time

    def set_opt_response_time(self, time: int):
        if self._opt_response_time is not None:
            raise ValueError("Optimal response time is already set!")
        self._opt_response_time = time

    @property
    def opt_response_time(self):
        if self._opt_response_time is None:
            raise ValueError("Optimal response time needs to be set!")
        return self._opt_response_time

    def set_simu_response_time(self, time: int):
        if self._simu_response_time is not None:
            raise ValueError("Simulated response time is already set!")
        self._simu_response_time = time

    @property
    def simu_response_time(self):
        if self._simu_response_time is None:
            raise ValueError("Simulated response time needs to be set!")
        return self._simu_response_time

    def to_dict(self):
        return {'event_type': self.event_type,
                'cell_id': self.cell_id,
                'response_unit': self.response_unit,
                'start_time': self.start_time,
                'opt_response_time': self.opt_response_time,
                'simu_response_time': self.simu_response_time}


class ResponseUnitImpl(ResponseUnit):

    def __init__(self, vehicles: list[dict]):
        self._vehicles = vehicles

    def get_veh_types(self):
        max_dur = self.get_max_dur()
        return [(veh['veh_type'], max_dur) for veh in self._vehicles]

    def get_max_dur(self):
        return max([veh['dur'] for veh in self._vehicles])


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


class MissionContainerImpl(MissionContainer):
    def __init__(self, missions: list[Mission]):
        self._missions = missions

    @property
    def missions(self):
        return self._missions

    def add_response_unit(self):
        for mission in self.missions:
            mission.set_response_unit(mission.event_type.sample_response_unit())

    def add_start_time(self, T_start, T_end):
        for mission in self.missions:
            mission.set_start_time(mission.event_type.sample_start_time(T_start, T_end))
        self.sort_by_start_time()

    def sort_by_start_time(self):
        self._missions.sort(key=lambda m: m.start_time)

    def save(self, output_file):
        pd.DataFrame([m.to_dict() for m in self.missions]).to_csv(output_file, index=False)


class VehicleDataTableImpl(VehicleDataTable):

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


class VehicleContainerImpl(VehicleContainer):

    def __init__(self, vehicles: list[Vehicle]):
        self.vehicles = self._structure_vehicles(vehicles)

    @staticmethod
    def _structure_vehicles(vehicles: list[Vehicle]) -> dict:
        vehicles_structured = defaultdict(list)
        for v in vehicles:
            vehicles_structured[v.type].append(v)
        return vehicles_structured

    def get_vehicle(self, veh_type: str, cell_id: str, travel_time_model: TravelTimeModel,
                    energy_table: VehicleDataTable):
        max_speed = energy_table.get_max_speed(veh_type)
        return min(self.vehicles[veh_type],
                   key=lambda veh: veh.next_avail_time + travel_time_model.get_travel_time(max_speed,
                                                                                           veh.station,
                                                                                           cell_id))

    def get_opt_arrival_time(self, veh_type: str, mission: Mission, travel_time_model: TravelTimeModel,
                             energy_table: VehicleDataTable):
        max_speed = energy_table.get_max_speed(veh_type)
        cell_id = mission.cell_id

        vehicle = min(self.vehicles[veh_type], key=lambda veh: travel_time_model.get_travel_time(max_speed,
                                                                                                 veh.station,
                                                                                                 cell_id))
        tt = travel_time_model.get_travel_time(max_speed, vehicle.station, cell_id)
        return mission.start_time + tt

    def simulate_mission(self, mission, travel_time_model, energy_table):
        """Response time is decided by the latest arrival time of the needed vehicles (response unit)."""
        latest_arrival_opt = 0
        latest_arrival_simu = 0

        for (veh_type, dur) in mission.response_unit.get_veh_types():
            vehicle = self.get_vehicle(veh_type, mission.cell_id, travel_time_model, energy_table)

            arrival_time_opt = self.get_opt_arrival_time(veh_type, mission, travel_time_model, energy_table)
            arrival_time_simu = self._simulate_movement(mission, vehicle, energy_table, travel_time_model, dur)

            if arrival_time_simu > latest_arrival_simu:
                latest_arrival_simu = arrival_time_simu

            if arrival_time_opt > latest_arrival_opt:
                latest_arrival_opt = arrival_time_opt

        mission.set_opt_response_time(latest_arrival_opt)
        mission.set_simu_response_time(latest_arrival_simu)

    def _simulate_movement(self, mission, vehicle, energy_table, travel_time_model, dur):
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
