import pytest
import pandas as pd
import load_data

"""
Loading Tests are as follows:
1) 
"""


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

def test_teacher_info_colums_length():
    assert len(load_data.load_campus_data()[1]) == 4


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
def test_coourse_info_colums_length():
    assert len(load_data.load_campus_data()[4]) == 6


def test_course_info():
    course_info_headers = ['course_id', 'priority', 'duration', 't1', 't2', 't3']
    assert course_info_headers == list(load_data.load_campus_data()[2])


def test_course_info_data_type_course_id():
    assert load_data.load_campus_data()[2]['course_id'].dtypes == 'int64'


def test_course_info_data_type_priority():
    assert load_data.load_campus_data()[2]['priority'].dtypes == 'int64'


def test_course_info_data_type_duration():
    assert load_data.load_campus_data()[2]['duration'].dtypes == 'int64'


def test_course_info_data_type_t1():
    assert load_data.load_campus_data()[2]['t1'].dtypes == 'object'


def test_course_info_data_type_t2():
    assert load_data.load_campus_data()[2]['t1'].dtypes == 'object'


def test_course_info_data_type_t3():
    assert load_data.load_campus_data()[2]['t1'].dtypes == 'object'


# Classroom
def test_classroom_info_colums_length():
    assert len(load_data.load_campus_data()[3]) == 3


def test_classroom_info():
    classroom_info_headers = ['classroom_id', 'area', 'ventilation_rate']
    assert classroom_info_headers == list(load_data.load_campus_data()[3])


def test_course_info_data_type_classroom_id():
    assert load_data.load_campus_data()[3]['classroom_id'].dtypes == 'int64'


def test_course_info_data_type_area():
    assert load_data.load_campus_data()[3]['area'].dtypes == 'int64'


def test_course_info_data_type_ventilation_rate():
    assert load_data.load_campus_data()[3]['ventilation_rate'].dtypes == 'int64'


# Community

def test_community_info():
    community_info_headers = ['week_number', 'community_risk', 'shutdown']
    assert community_info_headers == list(load_data.load_campus_data()[4])


def test_community_info_colums_length():
    assert len(load_data.load_campus_data()[4]) == 3


def test_community_info_data_type_week_number():
    assert load_data.load_campus_data()[4]['week_number'].dtypes == 'int64'


def test_community_info_data_type_community_risk():
    assert load_data.load_campus_data()[4]['community_risk'].dtypes == 'float64'


def test_community_info_data_type_shutdown():
    assert load_data.load_campus_data()[4]['shutdown'].dtypes == 'int64'


# Data Consistency

def test_course_id_in_student_teacher():
    all_student_courses = load_data.load_campus_data()[0][['c1', 'c2', 'c3', 'c4']].fillna(0).astype(
        'int64').values.tolist()  # Ensure that the final dataframe contains ints and not a mix of ints and floats
    all_teacher_courses = load_data.load_campus_data()[1][['c1', 'c2', 'c3']].fillna(0).astype('int64').values.tolist()
    all_courses = load_data.load_campus_data()['course_id'].fillna(0).astype('int64').values.tolist()
    all_courses_list = []
    student_check = []
    teacher_check = []

    # Append only values of the array object in courses list
    for i in all_courses:
        all_courses_list.append(i[0])

    # Append binary values to student check
    for courses in all_student_courses:
        check = all(item in all_courses_list for item in courses)
        if check is True:
            student_check.append(1)
        else:
            student_check.append(0)

    # Append binary values to techer check
    for courses in all_teacher_courses:
        check = all(item in all_courses_list for item in courses)
        if check is True:
            teacher_check.append(1)
        else:
            teacher_check.append(0)

    assert student_check == teacher_check

