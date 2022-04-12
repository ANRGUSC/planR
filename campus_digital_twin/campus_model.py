"""In this module is a general campus model is given.

Usage example:

campus = CampusModel()
students = campus.total_students()

"""
import os
from collections import Counter
import pandas as pd
import numpy as np


class CampusModel:
    """
    When created, the campus model object uses the default csv files, unless specified to extract 

      has the following key variables:

    Key Variables:
        student_df: dataframe
            with columns: [student_id, initial_infection,
            course1, course2, course3, course4]

        course_df: dataframe
            with columns: [course_id, priority, duration, t
            imeslot1, timeslot2, timeslot3]

        community_df: dataframe
            with columns: [week, community_risk_value]
    """
    counter = 0

    def __init__(self, student_df=None, course_df=None, community_df=None):
        rules = [student_df is None, course_df is None, community_df is None]
        if all(rules):
            self.student_df = pd.read_csv(
                open(os.path.dirname(os.path.realpath(__file__)) +
                     '/../input_files/student_info.csv'),
                error_bad_lines=False)

            self.course_df = pd.read_csv(
                open(os.path.dirname(os.path.realpath(__file__)) +
                     '/../input_files/course_info.csv'),
                error_bad_lines=False)
            self.classroom_df = pd.read_csv(open(os.path.dirname(os.path.realpath(__file__)) +
                                                 '/../input_files/classroom_info.csv'),
                                            error_bad_lines=False)
            self.community_df = pd.read_csv(
                open(os.path.dirname(os.path.realpath(__file__)) +
                     '/../input_files/community_info.csv'),
                error_bad_lines=False)

        else:
            self.student_df = student_df
            self.community_df = community_df
            self.course_df = course_df

        CampusModel.counter += 1

    def initial_community_risk(self):
        """Retrieve the community risk values
        Args: None
        Return: A list whose elements are the community risk values and index is the week number.

        """
        initial_community_risk = self.community_df['community_risk'].tolist()
        return initial_community_risk

    def student_initial_infection_status(self):
        """Count infected students from student_df.

        Retrieve the student infection status from the dataframe.

        Args: None
        Returns:
            A dict with keys 1 and 0 and whose value is the count of students.
            1 represents infected and 0 represents uninfected

        """
        initial_infection = self.student_df['initial_infection']
        initial_infection_list = initial_infection.tolist()
        distinct_list = Counter(initial_infection_list)
        return dict(distinct_list)

    def number_of_students_per_course(self):
        """Count the number of students per course.

        Converts the student_df to an array to
        count the number of students in a given course.

        Args: None

        Returns:
            1. A list with elements as total students and index are courses
            2. A list of list where the nested list contains the student_ids
            3. A dict whose keys are course_ids and values are student_ids

        """
        student_df = self.student_df[['c1']]

        student_course_array = student_df.to_numpy().astype(int)
        #print("student_df:", student_df)
        unique, frequency = np.unique(student_course_array, return_counts=True)
        student_course_dict = {}
        #print("unique:",unique)
        for course in unique:
            student_course_dict[course] = []
        for index, row in student_df.iterrows():
            student_course_dict[row['c1']].append(index)
            # student_course_dict[row['c2']].append(index)
            # student_course_dict[row['c3']].append(index)
        #print("student_course dict: ",student_course_dict)
        student_course_dict.pop(-1, None)
        frequency_list = []
        for course in student_course_dict:
            # remove if to consider more than one 'courses'
            if course == 0:
                frequency_list.append(len(student_course_dict[course]))
            else:
                break

        #print("student list", frequency_list)

        return frequency_list, unique, student_course_dict, frequency

    def number_of_infected_students_per_course(self):
        """Count the number of infected students per course.

        From the initial infection list, get the number of 1's.

        Return:
            1. A list whose elements are the total number of infected students per course
        """
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
            if course == 0:
                total_infected_students = len(students)
                infected_students_per_course_list.append(total_infected_students)
            else:
                break

        return infected_students_per_course_list

    def percentage_of_infected_students(self):
        """Calculate the percentage of infected students per course.
        Args: None

        Return:
            The percentage of infected students as a list whose index represents the course_id and
            elements are the percentages.

        """
        all_students = self.number_of_students_per_course()[0]
        infected_students = self.number_of_infected_students_per_course()
        percentage_of_infected_students = []
        for index, value in enumerate(all_students):
            percentage = int((int(infected_students[index]) / value) * 100)
            percentage_of_infected_students.append(percentage)

        return percentage_of_infected_students

    def get_max_weeks(self):
        """Get the number of weeks.

        Returns:
            weeks: Type(int)
        """
        return len(self.initial_community_risk())

    def total_courses(self):
        """Get the total courses
        Returns:
            total courses
        """
        return self.course_df.shape[0]
