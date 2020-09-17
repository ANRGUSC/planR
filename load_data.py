import os
import pandas as pd


def load_campus_data():
    # TODO: Update to this method to be more generic
    """
     # Get path to directory
     # Search for files that match names provided in a list
     # Return error for any missing files

    :return: dataframes for student, teacher, course, classroom and community
    """

    student_df = pd.read_csv(open('sampleInputFiles/student_info.csv'))
    teacher_df = pd.read_csv(open('sampleInputFiles/teacher_info.csv'))
    course_df = pd.read_csv(open('sampleInputFiles/course_info.csv'))
    classroom_df = pd.read_csv(open('sampleInputFiles/classroom_info.csv'))
    community_df = pd.read_csv(open('sampleInputFiles/community_info.csv'))

    return student_df, teacher_df, course_df, classroom_df, community_df


print(load_campus_data()[0]['student_id'].dtypes)
