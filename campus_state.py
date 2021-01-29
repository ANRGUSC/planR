class CampusState:

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

    def set_time(self, epochs):
        """
        week by week increament
        :type epochs: int
        """
        self.time = epochs

    def get_observation(self):
        observation = {
            'SS': self.student_status,
            'TS': self.teacher_status,
            'CQS': self.course_quarantine_status,
            'shutdown': self.shut_down,
            'CR': self.community_risk,
            'time': self.time
        }

        return observation

    def update_with_action(self, action):
        self.course_operation_status = action

    def schedule_class(self):
        """
        method needs to
        :param num_week:
        :return: list of tuples (course, classroom_id)
        """

    def update_with_infection_models(self):
        return

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

    # def get_reward(self, previous_state):
    #     reward = 0
    #     return reward


