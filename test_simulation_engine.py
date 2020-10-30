import pytest
import load_data
import simulation_engine

CampusState = simulation_engine.create_campus_state()


def test_student_status():
    assert type(CampusState.student_status) == list


def test_teacher_status():
    assert type(CampusState.teacher_status) == list


def test_course_quarantine_status():
    assert type(CampusState.course_quarantine_status) == bool


def test_shut_down():
    assert type(CampusState.shut_down) == bool


def test_time_value():
    assert CampusState.time[0].get("num_weeks") > 0
