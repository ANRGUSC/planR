from collections import defaultdict
from campus_digital_twin import campus_model as cm
from campus_digital_twin import scheduler as scheduler
from campus_digital_twin import infection_modela as im
#from campus_digital_twin import observations as observations

testNumber  = 8

class CampusState():

    def __init__(self, initialized=False, student_status=None, teacher_status=None, course_quarantine_status=None,
                 shut_down=None, community_risk=None):
        self.initialized = initialized
        self.student_status = student_status
        self.teacher_status = teacher_status
        self.course_quarantine_status = course_quarantine_status
        self.shut_down = shut_down
        self.community_risk = community_risk
        self.course_operation_status = None
        self.classroom_schedule = None
        self.model = cm.CampusModel()

    # Getters
    def get_student_status(self):
        return self.student_status

    def get_teacher_status(self):
        return self.teacher_status

    def get_course_quarantine_status(self):
        return self.course_quarantine_status

    def get_shut_down(self):
        return self.shut_down

    def get_community_risk(self):
        return self.community_risk

    def get_time(self):
        return self.time

    # Setters

    def set_student_status(self, student_status):
        """
            :type student_status: list
        """
        self.student_status = student_status

    def set_teacher_status(self, teacher_status):
        """
            :type teacher_status: list
        """
        self.teacher_status = teacher_status

    def set_course_quarantine_status(self, course_quarantine_status):
        """
            :type course_quarantine_status: list
        """
        self.course_quarantine_status = course_quarantine_status

    def set_shut_down(self, shut_down):
        """
            :type shut_down: list
        """
        self.shut_down = shut_down

    def set_community_risk(self, community_risk):
        """
        :type community_risk: int
        """
        self.community_risk = community_risk

    def get_observation(self):
        observation = self.model.is_conflict()
      
        # #observation = observations.observations
        # observation = {
        #     'SS': self.student_status,
        #     'TS': self.teacher_status,
        #     'CQS': self.course_quarantine_status,
        #     'shutdown': self.shut_down,
        #     'CR': self.community_risk,
        # }

        return observation

    def get_schedule(self):
        room_capacity = self.model.room_capacity()
        students_per_course = self.model.number_of_students_per_course()
        courses_with_conflict = self.model.is_conflict()
        schedule = scheduler.CourseRoomScheduler(room_capacity, students_per_course, courses_with_conflict)
        return schedule.get_schedule()

    def update_with_action(self, action):
        # TODO: (Elizabeth)

        self.course_operation_status = action

    def update_with_infection_model(self):
        number_of_students_per_course = self.model.number_of_students_per_course()
        community_risk = self.model.community_default
        new_infection_model = im.InfectionModel(number_of_students_per_course, community_risk)

        return new_infection_model.get_infected_students()

    def update_with_class_infection_model(self):
        return

    def update_with_community_infection_model(self):
        return

    def update_with_campus_infection_model(self):
        return

    def update_with_community(self):
        return

    def update_with_government_mandate(self):
        return

    def update_with_quarantine(self):
        return

    # def update_all(self):
    #   self.update_with_infection_models()
    #   self.update_with_community_infection_model()
    #   self.update_with_community()
    #   self.update_with_quarantine()
    #   return

    def get_reward(self):
        reward = 0
        return reward
