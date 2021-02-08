import pytest
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
    assert type(campusState.shut_down) == list
    for item in campusState.shut_down:
        assert type(item) == bool


def test_observation():
    """
    Call get_observation function from campus class
    Test the each element is correct
    :return:
    """
    campusState.get_observation()
    return

