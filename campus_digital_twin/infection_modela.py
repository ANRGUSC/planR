class InfectionModel(object):
    def __init__(self, number_of_students_per_course, community_risk):
        self.number_of_students_per_course = number_of_students_per_course
        self.community_risk = community_risk

    def get_infected_students(self):
        infected_students = []

        for i in self.number_of_students_per_course:
            infected_students.append(i * self.community_risk)

        return infected_students