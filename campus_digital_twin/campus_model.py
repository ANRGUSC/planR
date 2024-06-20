import random
random.seed(100)
import csv
class CampusModel:
    def __init__(self, num_courses=1, students_per_course=100, max_weeks=400,
                 initial_infection_rate= round(random.uniform(0.2, 0.7), 1)): #round(random.uniform(0.2, 0.7), 1) also we could add CR here??
        # with open('aggregated_weekly_risk_levels.csv', 'r') as file:
        #     csv_reader = csv.reader(file)
        #     next(csv_reader)  # Skip the header row
        #     risk_levels = [float(row[1]) for row in csv_reader]
        self.num_courses = num_courses
        self.students_per_course = [students_per_course] * num_courses  # Example: Same number of students for each course
        # self.max_weeks = len
        self.max_weeks = max_weeks
        # Construct the list of initial infection rates for each course
        self.initial_infection_rate = [initial_infection_rate] * num_courses

        # Calculate the initial number of infected students per course based on the infection rate
        self.initial_infection = [int(rate * students) for rate, students in
                                  zip(self.initial_infection_rate, self.students_per_course)]

    def number_of_students_per_course(self):
        # Return the deterministically generated number of students for each course
        return self.students_per_course

    def get_max_weeks(self):
        # Return the number of weeks
        return self.max_weeks

    def get_initial_infection(self):
        # Return the initial number of infected students per course
        return self.initial_infection

