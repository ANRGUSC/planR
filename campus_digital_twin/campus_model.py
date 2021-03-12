# This is a class to load campus specific data and includes methods for processing
import pandas as pd
import itertools as it
import numpy as np
import os
import yaml
from collections import Counter


class CampusModel:
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
        return frequency.tolist()

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

# Example

new_campus = CampusModel()
print(new_campus.initial_shutdown())
print(new_campus.initial_community_risk())
print(new_campus.student_initial_infection_status())
print(new_campus.teacher_initial_infection_status())
print(new_campus.initial_course_quarantine_status())
print(new_campus.total_students())
print(new_campus.total_rooms())
print(new_campus.total_courses())
print(new_campus.number_of_students_per_course())
print(new_campus.room_capacity())
print(new_campus.is_conflict())