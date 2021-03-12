# This is a class to load campus specific data and includes methods for processing
import pandas as pd
import itertools as it
import numpy as np
import os
import yaml
from collections import Counter

class CampusModel:
    def __init__(self, student_df=None, teacher_df=None, course_df=None, classroom_df=None, community_df=None):
        # Private Input parameters
        self._student_df = student_df
        self._teacher_df = teacher_df
        self._course_df = course_df
        self._classroom_df = classroom_df
        self._community_df = community_df

        self.student_default = pd.read_csv(open(os.path.dirname(os.path.realpath(__file__))+'/../sampleInputFiles/student_info.csv'), error_bad_lines=False)
        self.teacher_default = pd.read_csv(open(os.path.dirname(os.path.realpath(__file__))+'/../sampleInputFiles/teacher_info.csv'), error_bad_lines=False)
        self.course_default = pd.read_csv(open(os.path.dirname(os.path.realpath(__file__))+'/../sampleInputFiles/course_info.csv'), error_bad_lines=False)
        self.classroom_default = pd.read_csv(open(os.path.dirname(os.path.realpath(__file__))+'/../sampleInputFiles/classroom_info.csv'), error_bad_lines=False)
        self.community_default = pd.read_csv(open(os.path.dirname(os.path.realpath(__file__))+'/../sampleInputFiles/community_info.csv'), error_bad_lines=False)

    def load_sim_params(self, params_yaml):
        with open(params_yaml, 'r') as stream:
            try:
                sim_params = yaml.safe_load(stream)
                # print(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

        return sim_params

    def search_sim_params(self, params_list_of_dict, search_string):
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

    def generate_infection_list(self, list_of_dict):
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

    def get_simulation_params(self):
        #    sim_params = load_sim_params('campus_digital_twin/simulator_params.yaml')
        #    sim_params = load_sim_params('/home/runner/planR-7/campus_digital_twin/simulator_params.yaml')
        sim_params = self.load_sim_params(os.path.dirname(os.path.realpath(__file__)) + '/simulator_params.yaml')
        student_status = self.generate_infection_list(self.search_sim_params(sim_params, 'students'))
        teacher_status = self.generate_infection_list(self.search_sim_params(sim_params, 'teachers'))
        course_quarantine_status = self.search_sim_params(sim_params, 'course')
        shut_down = list(self.search_sim_params(sim_params, 'shutdown')[0].values())[0]
        community_risk = list(self.search_sim_params(sim_params, 'community')[0].values())[0]

        return student_status, teacher_status, course_quarantine_status, shut_down, community_risk

    def total_students(self):

        return self.student_default.shape[0]

    def total_courses(self):
        return self.course_default.shape[0]

    def total_rooms(self):
        return self.classroom_default.shape[0]

    def number_of_students_per_course(self):
        student_df = self.student_default[['c1', 'c2', 'c3', 'c4']]
        student_course_array = student_df.to_numpy().astype(int)
        unique, frequency = np.unique(student_course_array, return_counts=True)
        return frequency.tolist()

    def room_capacity(self):
        const_area_per_student = 25  # sqft
        classroom_df = self.classroom_default['area'].tolist()
        classroom_df_list = [int(i / const_area_per_student) for i in classroom_df]
        return classroom_df_list

    def is_conflict(self):
        """
        returns: an array showing which two classes have a conflict
        """
        course_conflict_dict = {}
        courses_matrix = np.zeros((self.total_courses(), self.total_courses()), dtype=bool)
        for column in self.course_default[['t1', 't2', 't3']]:
            courses_list = self.course_default
            courses_pairs = [p for p in it.product(courses_list['course_id'], repeat=2)]
            for pair in courses_pairs:
                course_a_time = eval(courses_list.iloc[pair[0]][str(column)])
                course_b_time = eval(courses_list.iloc[pair[1]][str(column)])
                if(course_a_time == course_b_time):
                    courses_matrix[pair[0], pair[1]] = True

        return (courses_matrix)


