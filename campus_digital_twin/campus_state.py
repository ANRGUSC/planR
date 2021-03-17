from collections import defaultdict
from campus_digital_twin import campus_model as cm
from campus_digital_twin import scheduler as scheduler
from campus_digital_twin import infection_modela as im

# from campus_digital_twin import observations as observations

testNumber = 8

class CampusState():
    model = cm.CampusModel()

    def __init__(self, initialized=False, student_status=model.number_of_infected_students_per_course(),
                 teacher_status=model.teacher_initial_infection_status(),
                 course_quarantine_status=model.initial_course_quarantine_status(),
                 shut_down=model.initial_shutdown(), community_risk=model.initial_community_risk()):
        self.initialized = initialized
        self.student_status = student_status
        self.teacher_status = teacher_status
        self.course_quarantine_status = course_quarantine_status
        self.shut_down = shut_down
        #self.community_risk = community_risk
        self.community_risk = (sum(community_risk) / len(community_risk))
        self.course_operation_status = None
        self.classroom_schedule = None

    def get_course_infection_status(self):
        self.model.number_of_students_per_course()

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
        observation = (self.student_status, self.community_risk)

        # #observation = observations.observations
        # observation = {
        #     'SS': self.student_status,
        #     'TS': self.teacher_status,
        #     'CQS': self.course_quarantine_status,
        #     'shutdown': self.shut_down,
        #     'CR': self.community_risk,
        # }

        return observation

    def get_course_infection_model(self):
        return

    def update_with_action(self, action):
        room_capacity = self.model.room_capacity()
        students_per_course = self.model.number_of_students_per_course()[0]
        courses_with_conflict = self.model.is_conflict()
        schedule = scheduler.CourseRoomScheduler(room_capacity, students_per_course,
                                                 courses_with_conflict).get_schedule(action)
        self.update_all(self.community_risk)
        self.set_community_risk(self.community_risk)
        return schedule

    def update_with_infection_model(self, community_risk):
        number_of_students_per_course = self.student_status
        # community_risk = self.community_risk
        infected_students = im.get_infected_students(number_of_students_per_course, community_risk)
        self.student_status = infected_students

    def update_with_class_infection_model(self):
        pass

    def update_with_community_infection_model(self):
        pass

    def update_with_campus_infection_model(self):
        pass

    def update_with_government_mandate(self):
        pass

    def update_all(self, community_risk):
        self.update_with_infection_model(community_risk)

    def get_reward(self):
        current_infected_students = sum(self.student_status)
        limit = 0.8 * self.model.total_students()
        if current_infected_students > limit:
            reward = -1
        else:
            reward = 1
        return reward
