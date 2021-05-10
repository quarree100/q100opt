import pytest

from q100opt.buildings import Building


def test_building_init():
    Building()


def test_no_roof_data_error():
    house = Building()
    with pytest.raises(ValueError,
                       match="Please provide roof data for a pre-calulation"):
        house.precalc_pv_profiles()
