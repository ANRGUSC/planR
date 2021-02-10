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

    def is_conflict(self):
        """
        Need to consider duration.
        Also the course times don't have days. Need to update campus csv generator
        """

        courses_list = self.course_default
        courses_matrix = np.zeros((self.total_courses(), self.total_courses()), dtype=bool)
        courses_pairs = [p for p in it.product(courses_list['course_id'], repeat=2)]
        course_conflict_dict = {}
        for column in self.course_default[['t1','t2','t3']]:
            for i in np.nditer(courses_matrix, op_flags=['readwrite']):
                for j in courses_pairs:
                    if j[0] == j[1]:
                       courses_matrix[j[0], j[1]] = False
                    else:
                        course_a = self.course_default.at[j[0], column]
                        course_b = self.course_default.at[j[1], column]
                        if(course_a == course_b):
                            courses_matrix[j[0], j[1]] = True
                        else:
                            courses_matrix[j[0], j[1]] = False

            course_conflict_dict[column] = courses_matrix

        return course_conflict_dict
test_campus_model = CampusModel()
print(test_campus_model.is_conflict())
# print(test_campus_model.room_capacity())
# print(test_campus_model.courses_with_conflict())