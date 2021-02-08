# This is a class to load campus specific data and includes methods for processing
import pandas as pd
import itertools as it
import numpy as np

class CampusModel:
    def __init__(self, student_df=None, teacher_df=None, course_df=None, classroom_df=None, community_df=None):
        # Private Input parameters
        self._student_df = student_df
        self._teacher_df = teacher_df
        self._course_df = course_df
        self._classroom_df = classroom_df
        self._community_df = community_df
        self.class_status = None
        self.student_default = pd.read_csv(open('sampleInputFiles/student_info.csv'), error_bad_lines=False)
        self.teacher_default = pd.read_csv(open('sampleInputFiles/teacher_info.csv'), error_bad_lines=False)
        self.course_default = pd.read_csv(open('sampleInputFiles/course_info.csv'), error_bad_lines=False)
        self.classroom_default = pd.read_csv(open('sampleInputFiles/classroom_info.csv'), error_bad_lines=False)
        self.community_default = pd.read_csv(open('sampleInputFiles/community_info.csv'), error_bad_lines=False)

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
        return frequency.tolist()[1:]

    def room_capacity(self):
        const_area_per_student = 25  # sqft
        classroom_df = self.classroom_default['area'].tolist()
        classroom_df_list = [int(i / const_area_per_student) for i in classroom_df]
        return classroom_df_list

    def courses_with_conflict(self):
        courses_list = self.course_default['course_id'].tolist()
        t1_list = self.course_default['t1']
        t2_list = self.course_default['t2']
        t3_list = self.course_default['t3']
        t1_course_combinations_conflicts = []
        t2_course_combinations_conflicts = []
        t3_course_combinations_conflicts = []
        courses_combinations = it.combinations(courses_list, 2)
        for course_pair in courses_combinations:
            index_a = course_pair[0]
            index_b = course_pair[1]

            value_a = t1_list[index_a]
            value_b = t1_list[index_b]

            value_c = t2_list[index_a]
            value_d = t2_list[index_b]

            value_e = t3_list[index_a]
            value_f = t3_list[index_b]


            if value_a == value_b:
                there_is_a_conflict = True
                t1_course_combinations_conflicts.append(there_is_a_conflict)
            else:
                there_is_a_conflict = False
                t1_course_combinations_conflicts.append(there_is_a_conflict)

            if value_c == value_d:
                there_is_a_conflict = True
                t2_course_combinations_conflicts.append(there_is_a_conflict)
            else:
                there_is_a_conflict = False
                t2_course_combinations_conflicts.append(there_is_a_conflict)

            if value_e == value_f:
                there_is_a_conflict = True
                t3_course_combinations_conflicts.append(there_is_a_conflict)
            else:
                there_is_a_conflict = False
                t3_course_combinations_conflicts.append(there_is_a_conflict)

        return (t1_course_combinations_conflicts, t2_course_combinations_conflicts,
                t3_course_combinations_conflicts)

test_campus_model = CampusModel()
print(test_campus_model.number_of_students_per_course())
print(test_campus_model.room_capacity())
print(test_campus_model.courses_with_conflict())