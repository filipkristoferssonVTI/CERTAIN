import copy
import math

import pandas as pd
import pytest

from model.simulation.implementations import ResponseUnitImpl, EventTypeImpl, Mission, MissionContainerImpl, \
    TravelTimeModelImpl, Vehicle, VehicleContainerImpl, VehicleDataTableImpl


@pytest.fixture
def vehicles():
    return [
        Vehicle(id='1',
                station='S1',
                type='80',
                battery_level=100),
        Vehicle(id='2',
                station='S2',
                type='65',
                battery_level=200),
        Vehicle(id='3',
                station='S2',
                type='80',
                battery_level=100)
    ]


@pytest.fixture
def ru_vehicles():
    return ([{"veh_type": "80", "dur": 15},
             {"veh_type": "65", "dur": 20}],
            [{"veh_type": "80", "dur": 25},
             {"veh_type": "65", "dur": 30}])


@pytest.fixture
def response_units(ru_vehicles):
    veh_group_1, veh_group_2 = ru_vehicles
    return [
        ResponseUnitImpl(veh_group_1),
        ResponseUnitImpl(veh_group_2),
    ]


@pytest.fixture
def event_type(response_units):
    return EventTypeImpl('event_type_1', response_units)


@pytest.fixture
def missions(event_type):
    return [
        Mission(copy.deepcopy(event_type), 'C1'),
        Mission(copy.deepcopy(event_type), 'C2'),
    ]


@pytest.fixture
def od_matrix():
    return pd.DataFrame(
        {
            "cell_id": ["C1", "C1", "C2", "C2"],
            "station_id": ["S1", "S2", "S1", "S2"],
            "dist_m": [1200, 540, 800, 3100],
        }
    )


@pytest.fixture
def energy_table():
    return pd.DataFrame(
        {
            "battery_cap": [100, 200],
            "energy_need": [1, 1.5],
            "charge_time": [300, 400],
            "max_speed": [60, 60],
        },
        index=["80", "65"],
    )


def test_response_unit(response_units):
    ru = response_units[0]
    assert ru.get_veh_types() == [
        ("80", 20),  # same dur for both since we use max dur within the response unit
        ("65", 20),
    ]


def test_event_type(event_type, response_units):
    response_unit = event_type.sample_response_unit()
    assert any(
        response_unit.get_veh_types() == ru.get_veh_types()
        for ru in response_units
    )


def test_mission_container(missions, response_units):
    mc = MissionContainerImpl(missions)
    mc.add_response_unit()
    mc.add_start_time(0, 10)

    for m in mc.missions:
        assert 0 <= m.start_time <= 10
        assert any(
            m.response_unit.get_veh_types() == ru.get_veh_types()
            for ru in response_units
        )

    times = [m.start_time for m in mc.missions]
    assert times == sorted(times), f"Not sorted: {times}"


def test_travel_time_model(od_matrix):
    ttm = TravelTimeModelImpl(od_matrix)
    dist = ttm.get_distance('S1', 'C1')
    max_speed = 60
    assert dist == 1200
    assert ttm.get_travel_time(max_speed, 'S1', 'C1') == dist / max_speed


def test_vehicle_container_get_vehicle_get_opt_arrival_time(vehicles, od_matrix, energy_table, missions):
    vc = VehicleContainerImpl(vehicles)
    ttm = TravelTimeModelImpl(od_matrix)
    vdt = VehicleDataTableImpl(energy_table)
    mc = MissionContainerImpl(missions)

    mc.add_start_time(0, 10)
    mc.add_response_unit()
    mission = mc.missions[0]

    veh_types_needed = mission.response_unit.get_veh_types()

    for (veh_type, _) in veh_types_needed:
        vehicles = vc.vehicles[veh_type]  # every vehicle of the needed type
        selected_vehicle = None  # the vehicle with the min arrival time
        min_arrival_time = math.inf

        for v in vehicles:
            dist = ttm.get_distance(v.station, mission.cell_id)
            next_avail_time = v.next_avail_time
            max_speed = vdt.get_max_speed(veh_type)
            tt = math.ceil(dist / max_speed)
            arrival_time = next_avail_time + tt
            if arrival_time < min_arrival_time:
                min_arrival_time = arrival_time
                selected_vehicle = v

        veh = vc.get_vehicle(veh_type, mission.cell_id, ttm, vdt)
        assert veh == selected_vehicle

        opt_arrival_time = vc.get_opt_arrival_time(veh_type, mission, ttm, vdt)
        assert opt_arrival_time == mission.start_time + min_arrival_time - selected_vehicle.next_avail_time


def test_vehicle_container_simulate_mission(vehicles, od_matrix, energy_table, missions):
    vc = VehicleContainerImpl(vehicles)
    ttm = TravelTimeModelImpl(od_matrix)
    vdt = VehicleDataTableImpl(energy_table)
    mc = MissionContainerImpl(missions)

    mc.add_start_time(0, 10)
    mc.add_response_unit()
    mission = mc.missions[0]
    cell_id = mission.cell_id

    veh_types_needed = mission.response_unit.get_veh_types()
    selected_veh = [vc.get_vehicle(veh_type, cell_id, ttm, vdt) for (veh_type, _) in veh_types_needed]
    selected_veh_dur = [dur for (_, dur) in veh_types_needed]

    opt_arrival_times = [vc.get_opt_arrival_time(v.type, mission, ttm, vdt) for v in selected_veh]
    next_avail_times = [v.next_avail_time for v in selected_veh]
    travel_times = [ttm.get_travel_time(vdt.get_max_speed(v.type), v.station, cell_id) for v in selected_veh]
    simu_arrival_times = [max(nat, mission.start_time) + tt for nat, tt in zip(next_avail_times, travel_times)]

    vc.simulate_mission(mission, ttm, vdt)

    assert mission.opt_response_time == max(opt_arrival_times)
    assert mission.simu_response_time == max(simu_arrival_times)

    for veh, nat, tt, dur in zip(selected_veh, next_avail_times, travel_times, selected_veh_dur):
        assert veh.next_avail_time == max(mission.start_time, nat) + tt + dur + tt  # tt twice since we need to return
