# This is a class to load campus specific data and includes methods for processing
import pandas as pd


class CampusModel:
    # The campus is modeled as a
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
        self.classroom_defaut = pd.read_csv(open('sampleInputFiles/classroom_info.csv'), error_bad_lines=False)
        self.community_default = pd.read_csv(open('sampleInputFiles/community_info.csv'), error_bad_lines=False)


myCampus = CampusModel()
print(myCampus.student_default)