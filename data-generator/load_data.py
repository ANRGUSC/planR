import pandas as pd

def load_campus_data():
    # TODO: Update to this method to be more generic
    # add argument
    """
     # Receive path to directory, file names
     # Return error for any missing files

    :return: dataframes for student, teacher, course, classroom and community
    """

    student_df = pd.read_csv(open('../sampleInputFiles/student_info.csv'), error_bad_lines=False)
    teacher_df = pd.read_csv(open('../sampleInputFiles/teacher_info.csv'), error_bad_lines=False)
    course_df = pd.read_csv(open('../sampleInputFiles/course_info.csv'), error_bad_lines=False)
    classroom_df = pd.read_csv(open('../sampleInputFiles/classroom_info.csv'), error_bad_lines=False)
    community_df = pd.read_csv(open('../sampleInputFiles/community_info.csv'), error_bad_lines=False)

    return student_df, teacher_df, course_df, classroom_df, community_df

