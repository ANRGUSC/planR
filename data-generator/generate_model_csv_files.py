"""
Portion A: Generates csv files based on user requirements

"""
import random
import csv
import yaml
import os


def load_sim_params(params_yaml):
    with open(params_yaml, 'r') as stream:
        try:
            sim_params = yaml.safe_load(stream)
            # print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    return sim_params

def search_sim_params(params_list_of_dict, search_string):
    """
    :param params_list_of_dict:
    :param search_string:
    :return: list
    """
    data_list = []
    for i in params_list_of_dict:
        [[key, value]] = i.items()
        my_list = key.split("_")
        if search_string in my_list:
            data_list.append(i)

    return data_list
#
#
def generate_infection_list(list_of_dict):
    """
    :param -> list_of_dict:
    :return:
    """
    total = list_of_dict[0].values
    status_list = []
    for status in list_of_dict[1:]:
        for key, value in status.items():
            if value == 0 or value < 0:
                pass
            else:
                infection_status = [key] * value
                status_list = status_list + infection_status

    return status_list



def convert_to_twenty_four_hours(value_int):
    if len(str(value_int)) > 1:
        return str(value_int) + "0" + "0"
    else:
        return "0" + str(value_int) + "0" + "0"


def create_csv_files():
    # Get user keyboard input
    sim_params = load_sim_params(os.path.dirname(os.path.realpath(__file__))+'/simulator_params.yaml')

    totals = search_sim_params(sim_params, 'num')
    total_students = totals[0].get('num_students')
    total_teachers = totals[1].get('num_teacher')
    total_classrooms = totals[2].get('num_classes')
    total_courses = totals[3].get('num_courses')
    weeks_of_operation = totals[4].get('num_weeks')

    # # This will be mapped to cav
    student_info = []
    teacher_info = []
    course_info = []
    classroom_info = []
    community_info = []

    # # Students
    student_columns = ['student_id', 'initial_infection', 'c1', 'c2', 'c3', 'c4']
    for student_id in range(0, total_students):
        course_list = random.sample([0, 1, 2, 3], 4)
        c1 = course_list[0]
        c2 = course_list[1]
        c3 = course_list[2]
        c4 = course_list[3]
        student_info_rows = {'student_id': student_id, 'initial_infection': random.getrandbits(1), 'c1': c1,
                             'c2': c2, 'c3': c3, 'c4': c4}
        student_info.append(student_info_rows)

    # # Teachers
    teachers_columns = ['teacher_id', 'c1', 'c2', 'c3']
    for teacher_id in range(0, total_teachers):
        course_list = random.sample([0, 1, 2, 3], 3)
        c1 = course_list[0]
        c2 = course_list[1]
        c3 = course_list[2]
        teacher_info_rows = {'teacher_id': teacher_id, 'c1': c1, 'c2': c2, 'c3': c3}
        teacher_info.append(teacher_info_rows)

    # # Courses
    course_columns = ['course_id', 'priority', 'duration', 't1', 't2', 't3']
    for course_id in range(0, total_courses):
        days = ['M', 'T', 'W', 'TH', 'F']
        t1 = convert_to_twenty_four_hours(random.randrange(8, 17))
        t2 = convert_to_twenty_four_hours(random.randrange(8, 17))
        t3 = convert_to_twenty_four_hours(random.randrange(8, 17))
        priority = random.getrandbits(1)
        duration = int(random.randrange(90, 120))
        course_info_rows = {'course_id': course_id, 'priority': priority,
                            'duration': duration, 't1': (random.choice(days), t1),
                            't2': (random.choice(days), t2), 't3': (random.choice(days), t3)}
        course_info.append(course_info_rows)
    #
    # # Classrooms
    classroom_columns = ['classroom_id', 'area', 'ventilation_rate']
    for classroom_id in range(0, total_classrooms):
        area = int(random.randrange(150, 520))
        ventilation_rate = int(random.randrange(1, 10))
        classroom_info_rows = {'classroom_id': classroom_id, 'area': area,
                               'ventilation_rate': ventilation_rate}
        classroom_info.append(classroom_info_rows)
    #
    # # Community
    community_columns = ['week_number', 'community_risk', 'shutdown']
    for week in range(0, weeks_of_operation):
        shutdown = random.getrandbits(1)
        community_risk = random.uniform(0,1)
        community_info_rows = {'week_number': week, 'community_risk': community_risk,
                               'shutdown': shutdown}
        community_info.append(community_info_rows)

    # # Save data generated to csv file for use by simulator
    student_csv_file = "../sampleInputFiles/student_info.csv"
    try:
        with open(student_csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=student_columns)
            writer.writeheader()
            for data in student_info:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    teacher_csv_file = "../sampleInputFiles/teacher_info.csv"
    try:
        with open(teacher_csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=teachers_columns)
            writer.writeheader()
            for data in teacher_info:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    course_csv_file = "../sampleInputFiles/course_info.csv"
    try:
        with open(course_csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=course_columns)
            writer.writeheader()
            for data in course_info:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    classroom_csv_file = "../sampleInputFiles/classroom_info.csv"
    try:
        with open(classroom_csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=classroom_columns)
            writer.writeheader()
            for data in classroom_info:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    community_csv_file = "../sampleInputFiles/community_info.csv"
    try:
        with open(community_csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=community_columns)
            writer.writeheader()
            for data in community_info:
                writer.writerow(data)
    except IOError:
        print("I/O error")


create_csv_files()


