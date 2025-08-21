from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This import is for mypy/pyright only; it won't run at runtime.
    from model.simulation.implementations import Mission, Vehicle


class ResponseUnit(ABC):
    @abstractmethod
    def get_veh_types(self) -> list[tuple[str, int]]:
        """Returns a list of (veh_type, dur) tuples."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(veh_types='{self.get_veh_types()}')"


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


class MissionContainer(ABC):

    @property
    @abstractmethod
    def missions(self) -> list[Mission]:
        pass

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


class VehicleDataTable(ABC):
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


class VehicleContainer(ABC):

    @abstractmethod
    def simulate_mission(self, mission: Mission, travel_time_model: TravelTimeModel, energy_table: VehicleDataTable):
        pass

    @abstractmethod
    def _simulate_movement(self, mission: Mission, vehicle: Vehicle, energy_table: VehicleDataTable,
                           travel_time_model: TravelTimeModel, dur: int) -> int:
        pass


class RegrModel(ABC):
    @abstractmethod
    def gen_missions(self, event_type: EventType) -> list[Mission]:
        """Generates a number of missions of a given event type."""
        pass
