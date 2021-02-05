import pytest
import load_data
import simulation_engine

"""
Tests:
for student and teacher status
check all the elements and not just the object
check the length of list
"""
campusState = simulation_engine.create_campus_state()


def test_student_status():
    assert type(campusState.student_status) == list
    for item in campusState.student_status:
        assert type(item) == str


def test_teacher_status():
    assert type(campusState.teacher_status) == list
    for item in campusState.teacher_status:
        assert type(item) == str


def test_course_quarantine_status():
    assert type(campusState.course_quarantine_status) == list


def test_shut_down():
    assert type(campusState.shut_down) == bool


def test_time_value():
    assert campusState.time[0].get("num_weeks") > 0

def test_observation():
    """
    Call get_observation function from campus class
    Test the each element is correct
    :return:
    """
    return
