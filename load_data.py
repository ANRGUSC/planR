import os, sys
import pandas as pd



def load_campus_data(campus_real_files):
    # TODO: Update to this method to be more generic
    """
     # Receive path to directory, file names
     # Return error for any missing files

    :return: dataframes for student, teacher, course, classroom and community
    """

    student_df = pd.read_csv(open('sampleInputFiles/student_info.csv'))
    teacher_df = pd.read_csv(open('sampleInputFiles/teacher_info.csv'))
    course_df = pd.read_csv(open('sampleInputFiles/course_info.csv'))
    classroom_df = pd.read_csv(open('sampleInputFiles/classroom_info.csv'))
    community_df = pd.read_csv(open('sampleInputFiles/community_info.csv'))

    return student_df, teacher_df, course_df, classroom_df, community_df



