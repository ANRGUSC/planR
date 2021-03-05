class InfectionModel(object):
    def __init__(self, number_of_students_per_course, community_risk):
        self.number_of_students_per_course = number_of_students_per_course
        self.community_risk = community_risk

    def get_classroom_infection_model(self):
        return