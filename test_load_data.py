import pytest
import pandas as pd
import load_data


# Student

def test_student_info_colums_length():
    assert len(load_data.load_campus_data()[0]) == 6


def test_student_info_colums_labels():
    student_info_headers = ['student_id', 'initial_infection', 'c1', 'c2', 'c3', 'c4']
    assert student_info_headers == list(load_data.load_campus_data()[0])


def test_student_info_data_type_student_info():
    assert load_data.load_campus_data()[0]['student_id'].dtypes == 'int64'


def test_student_info_data_type_infection_info():
    assert load_data.load_campus_data()[0]['initial_infection'].dtypes == 'int64'


def test_student_info_data_type_c1():
    assert load_data.load_campus_data()[0]['c1'].dtypes == 'int64'


def test_student_info_data_type_c2():
    assert load_data.load_campus_data()[0]['c2'].dtypes == 'int64'


def test_student_info_data_type_c3():
    assert load_data.load_campus_data()[0]['c3'].dtypes == 'int64'


def test_student_info_data_type_c4():
    assert load_data.load_campus_data()[0]['c4'].dtypes == 'int64'


# Teacher

def test_teacher_info():
    teacher_info_headers = ['teacher_id', 'c1', 'c1', 'c2', 'c3']
    assert teacher_info_headers == list(load_data.load_campus_data()[1])


def test_teacher_info_data_type_teacher_id():
    assert load_data.load_campus_data()[1]['teacher_id'].dtypes == 'int64'


def test_teacher_info_data_type_c1():
    assert load_data.load_campus_data()[1]['c1'].dtypes == 'int64'


def test_teacher_info_data_type_c2():
    assert load_data.load_campus_data()[1]['c2'].dtypes == 'int64'


def test_teacher_info_data_type_c3():
    assert load_data.load_campus_data()[1]['c3'].dtypes == 'int64'


# Course

def test_course_info():
    course_info_headers = ['course_id', 'priority', 'duration', 't1', 't2', 't3']
    assert course_info_headers == list(load_data.load_campus_data()[2])


# Classroom
def test_classroom_info():
    classroom_info_headers = ['classroom_id', 'area', 'ventilation_rate']
    assert classroom_info_headers == list(load_data.load_campus_data()[3])


# Community
def test_community_info():
    community_info_headers = ['week_number', 'community_risk', 'shutdown']
    assert community_info_headers == list(load_data.load_campus_data()[4])
