import os, sys
import pandas as pd

ret = os.access("sampleInputFiles", os.F_OK)
print("F_OK - return value %s" % ret)


def load_campus_data():
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


print(load_campus_data('sampleInputFiles')[0][['c1', 'c2', 'c3', 'c4']])

student_courses = load_campus_data('sampleInputFiles')[0][['c1', 'c2', 'c3', 'c4']].fillna(0)

student_courses_np = student_courses.astype('int64')

all_courses = load_campus_data('sampleInputFiles')[2][['course_id']].fillna(0)



student_courses_np_list = student_courses_np.values.tolist()
all_courses_list = all_courses.values.tolist()

flat_student_courses_list = [item for sublist in student_courses_np_list for item in sublist]

print(set(flat_student_courses_list))
# all_courses_flist = []
# student_check = []
# for i in all_courses_list:
#     all_courses_flist.append(i[0])
#
# for courses in student_courses_np_list:
#     check = all(item in all_courses_flist for item in courses)
#     if check is True:
#         student_check.append(1)
#     else:
#         student_check.append(0)
#
# print (student_check)
#
#
