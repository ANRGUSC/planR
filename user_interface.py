"""
Portion A: Generates csv files based on user requirements

"""
import random
import csv
import numpy as np

print("Hello, welcome to SafeCampusRL")


def convert_to_twenty_four_hours(value_int):
    if len(str(value_int)) > 1:
        return str(value_int) + "0" + "0"
    else:
        return "0" + str(value_int) + "0" + "0"


def create_csv_files():
    # Get user keyboard input
    total_students = int(input("Students: "))
    total_teachers = int(input("Teachers: "))
    total_courses = int(input("Courses: "))
    total_classrooms = int(input("Classrooms: "))
    weeks_of_operation = int(input("Weeks: "))

    # This will be mapped to cav
    student_info = []
    teacher_info = []
    course_info = []
    classroom_info = []
    community_info = []

    # Students
    #TODO: This needs to be updated such that a student cannot have more than 2 of the same courses.
    student_columns = ['student_id', 'initial_infection', 'c1', 'c2', 'c3', 'c4']
    for student_id in range(0, total_students):
        c1 = random.randrange(0, total_courses)
        c2 = random.randrange(0, total_courses)
        c3 = random.randrange(0, total_courses)
        c4 = random.randrange(0, total_courses)
        student_info_rows = {'student_id': student_id, 'initial_infection': random.getrandbits(1), 'c1': c1,
                             'c2': c2, 'c3': c3, 'c4': c4}
        student_info.append(student_info_rows)

    # Teachers
    teachers_columns = ['teacher_id', 'c1', 'c2', 'c3']
    for teacher_id in range(0, total_teachers):
        c1 = random.randrange(0, total_courses)
        c2 = random.randrange(0, total_courses)
        c3 = random.randrange(0, total_courses)
        teacher_info_rows = {'teacher_id': teacher_id, 'c1': c1, 'c2': c2, 'c3': c3}
        teacher_info.append(teacher_info_rows)

    # Courses
    course_columns = ['course_id', 'priority', 'duration', 't1', 't2', 't3']
    for course_id in range(0, total_courses):
        days = ['M', 'T', 'W', 'TH', 'F']
        t1 = convert_to_twenty_four_hours(random.randrange(8, 17))
        t2 = convert_to_twenty_four_hours(random.randrange(8, 17))
        t3 = convert_to_twenty_four_hours(random.randrange(8, 17))
        priority = random.getrandbits(1)
        duration = int(random.randrange(90, 120))
        course_info_rows = {'course_id': course_id, 'priority': priority,
                            'duration': duration, 't1': t1, 't2': t2, 't3': t3}
        course_info.append(course_info_rows)

    # Classrooms
    classroom_columns = ['classroom_id', 'area', 'ventilation_rate']
    for classroom_id in range(0, total_classrooms):
        area = int(random.randrange(300, 1200))
        ventilation_rate = int(random.randrange(1, 10))
        classroom_info_rows = {'classroom_id': classroom_id, 'area': area,
                               'ventilation_rate': ventilation_rate}
        classroom_info.append(classroom_info_rows)

    # Community
    community_columns = ['week_number', 'community_risk', 'shutdown']
    for week in range(0, weeks_of_operation):
        shutdown = random.getrandbits(1)
        community_risk = np.arange(0, 0.9, 0.01)
        community_info_rows = {'week_number': week, 'community_risk': community_risk,
                               'shutdown': shutdown}
        community_info.append(community_info_rows)

    # Save data generated to csv file for use by simulator
    student_csv_file = "sampleInputFiles/student_info.csv"
    try:
        with open(student_csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=student_columns)
            writer.writeheader()
            for data in student_info:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    teacher_csv_file = "sampleInputFiles/teacher_info.csv"
    try:
        with open(teacher_csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=teachers_columns)
            writer.writeheader()
            for data in teacher_info:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    course_csv_file = "sampleInputFiles/course_info.csv"
    try:
        with open(course_csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=course_columns)
            writer.writeheader()
            for data in course_info:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    classroom_csv_file = "sampleInputFiles/classroom_info.csv"
    try:
        with open(classroom_csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=classroom_columns)
            writer.writeheader()
            for data in classroom_info:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    community_csv_file = "sampleInputFiles/community_info.csv"
    try:
        with open(community_csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=community_columns)
            writer.writeheader()
            for data in community_info:
                writer.writerow(data)
    except IOError:
        print("I/O error")


create_csv_files()


