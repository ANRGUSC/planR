import load_data

"""
Loading Tests are as follows:
1) 
"""


# Student

# TODO: Test course-id duplicates on the same row

# def test_student_course():

def test_student_info_colums_length():
    assert len(load_data.load_campus_data()[0]) == 6


def test_student_info_colums_labels():
    student_info_headers = ['student_id', 'initial_infection', 'c1', 'c2', 'c3', 'c4']
    assert student_info_headers == list(load_data.load_campus_data()[0])


def test_student_info_data_type_student_info():
    assert load_data.load_campus_data()[0]['student_id'].dtypes == 'int64'


# Add more infection status test to be 0-5 and non negative
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
    # Tests that all_student_courses is a subset of all_courses
    # test that all_teacher
    # Every course is in the student list
    # Every

    all_student_courses = load_data.load_campus_data()[0][['c1', 'c2', 'c3', 'c4']].fillna(0).astype(
        'int64').values.tolist()  # Ensure that the final dataframe contains ints and not a mix of ints and floats
    all_teacher_courses = load_data.load_campus_data()[1][['c1', 'c2', 'c3']].fillna(0).astype('int64').values.tolist()
    all_courses = load_data.load_campus_data()['course_id'].fillna(0).astype('int64').values.tolist()

    flat_student_courses_list = [item for sublist in all_student_courses for item in sublist]
    flat_teacher_courses_list = [item for sublist in all_teacher_courses for item in sublist]
    flat_courses_list = [item for sublist in all_courses for item in sublist]

    student_courses_set = set(flat_student_courses_list)
    teacher_courses_set = set(flat_teacher_courses_list)
    courses_set = set(flat_courses_list)

    assert (teacher_courses_set == student_courses_set) and (student_courses_set == courses_set)

    # Append only values of the array object in courses list
