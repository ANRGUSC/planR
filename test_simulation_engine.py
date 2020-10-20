import pytest
import load_data
import simulation_engine

CampusState = simulation_engine.create_campus_state()

"""
 Tests:
 ***** TODO:
 - student_status is a list of strings
 - entries are from the set of possible values
 - teacher_status is the same but length D
 - CampusState object with parameters at time 0 match .yaml file
 ***** END TODO
 
 test course_quarantine_status is a list of booleans of size C
 shutdown is a single boolean variable
 community risk is a float between 0 and 1
 time is a non negative integer

"""


def test_course_quarantine_status():
    assert type(CampusState.course_quarantine_status) == bool


def test_shut_down():
    assert type(CampusState.shut_down) == bool


def test_time():
    assert CampusState.time[0].get("num_weeks") > 0
