# This is a class to load campus specific data and includes methods for processing
import pandas as pd
import itertools as it
import numpy as np
import os
import yaml
from collections import Counter

class CampusModel:
    counter = 0
    def __init__(self, student_df=None, teacher_df=None, course_df=None, classroom_df=None, community_df=None):
        rules = [student_df is None, teacher_df is None, course_df is None, classroom_df is None, community_df is None]
        if all(rules):
            self.student_df = pd.read_csv(
                open(os.path.dirname(os.path.realpath(__file__)) + '/../sampleInputFiles/student_info.csv'),
                error_bad_lines=False)

            self.teacher_df = pd.read_csv(
                open(os.path.dirname(os.path.realpath(__file__)) + '/../sampleInputFiles/teacher_info.csv'),
                error_bad_lines=False)

            self.course_df = pd.read_csv(
                open(os.path.dirname(os.path.realpath(__file__)) + '/../sampleInputFiles/course_info.csv'),
                error_bad_lines=False)
            self.classroom_df = pd.read_csv(
                open(os.path.dirname(os.path.realpath(__file__)) + '/../sampleInputFiles/classroom_info.csv'),
                error_bad_lines=False)
            self.community_df = pd.read_csv(
                open(os.path.dirname(os.path.realpath(__file__)) + '/../sampleInputFiles/community_info.csv'),
                error_bad_lines=False)
        else:
            self.student_df = student_df
            self.teacher_df = teacher_df
            self.community_df = community_df
            self.classroom_df = classroom_df
            self.course_df = course_df
        CampusModel.counter += 1

    def student_initial_infection_status(self):
        initial_infection = self.student_df['initial_infection']
        initial_infection_list = initial_infection.tolist()
        distinct_list = Counter(initial_infection_list)
        return dict(distinct_list)

    def initial_course_quarantine_status(self):
        return []

    def initial_community_risk(self):
        initial_community_risk = self.community_df['community_risk'].tolist()
        return initial_community_risk

    def get_max_weeks(self):
        return len(self.initial_community_risk())

    def initial_shutdown(self):
        initial_shutdown = self.community_df['shutdown'].tolist()
        return initial_shutdown

    def teacher_initial_infection_status(self):
        return {}

    def total_students(self):

        return self.student_df.shape[0]

    def total_courses(self):
        return self.course_df.shape[0]

    def total_rooms(self):
        return self.classroom_df.shape[0]

    def number_of_students_per_course(self):
        student_df = self.student_df[['c1', 'c2', 'c3']]
        student_course_array = student_df.to_numpy().astype(int)
        unique, frequency = np.unique(student_course_array, return_counts=True)
        student_course_dict= {}
        for course in unique:
            student_course_dict[course] = []
        for index, row in student_df.iterrows():

            student_course_dict[row['c1']].append(index)
            student_course_dict[row['c2']].append(index)
            student_course_dict[row['c3']].append(index)

        student_course_dict.pop(-1, None)
        frequency_list = []
        for course in student_course_dict:
            frequency_list.append(len(student_course_dict[course]))

        return frequency_list, unique, student_course_dict

    def number_of_infected_students_per_course(self):
        infected_student_list = self.student_df['initial_infection'].tolist()
        students_per_course = self.number_of_students_per_course()[2]
        infected_student_ids = []
        infected_students_per_course = {}
        infected_students_per_course_list = []

        for course in students_per_course:
            infected_students_per_course[course] = []

        for index, value in enumerate(infected_student_list):
            if value == 1:
                infected_student_ids.append(index)

        for course, students in students_per_course.items():
            for student in infected_student_ids:
                if student in students:
                    infected_students_per_course[course].append(student)

        for course, students in infected_students_per_course.items():
            total_infected_students = len(students)
            infected_students_per_course_list.append(total_infected_students)

        return infected_students_per_course_list

    def percentage_of_infected_students_per_course(self):
        total_students_per_course = self.number_of_students_per_course()[0]
        total_infected_students = self.number_of_infected_students_per_course()
        percentage_of_infected_students = []
        for index, value in enumerate(total_students_per_course):
            percentage = int((int(total_infected_students[index])/value) * 100)
            percentage_of_infected_students.append(percentage)

        return percentage_of_infected_students

    def room_capacity(self):
        const_area_per_student = 25  # sqft
        classroom_df = self.classroom_df['area'].tolist()
        classroom_df_list = [int(i / const_area_per_student) for i in classroom_df]
        return classroom_df_list

    def is_conflict(self):
        """
        returns: an array showing which two classes have a conflict. True means there is a conflict. Note that the
        diagonal values of the matrix are True and ignored.
        """
        course_conflict_dict = {}
        courses_matrix = np.zeros((self.total_courses(), self.total_courses()), dtype=bool)
        for column in self.course_df[['t1', 't2', 't3']]:
            courses_list = self.course_df
            courses_pairs = [p for p in it.product(courses_list['course_id'], repeat=2)]
            for pair in courses_pairs:
                course_a_time = eval(courses_list.iloc[pair[0]][str(column)])
                course_b_time = eval(courses_list.iloc[pair[1]][str(column)])
                if (course_a_time == course_b_time):
                    courses_matrix[pair[0], pair[1]] = True

        return (courses_matrix)

