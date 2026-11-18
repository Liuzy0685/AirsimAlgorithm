import pytest
from control.swept_clearance import measured_command_clearance


def test_age_and_actual_duration_are_both_in_the_checked_path():
    # Measured motion travels .2m while data ages, then the issued command
    # travels .1m. A point at .4m must be only .1m from the sweep.
    assert measured_command_clearance([[.4,0,0]],[1,0,0],[1,0,0],.1,.2)==pytest.approx(.1)


def test_turn_is_two_segments_instead_of_a_three_second_straight_line():
    d=measured_command_clearance([[1.,0.,0.]],[.35,0,0],[0,.35,0],.15,.2)
    assert d==pytest.approx(.93)


def test_nearby_obstacle_is_never_ignored():
    assert measured_command_clearance([[.1,0,0]],[0,0,0],[0,.35,0],.15,0)==pytest.approx(.1)
    with pytest.raises(ValueError):measured_command_clearance([], [0,0,0],[0,0,0],.15,0)
