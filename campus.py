import load_data


# student_status = load_data.load_campus_data()[0]['infection']

class CampusState:

    def __init__(self, student_status, teacher_status, course_quarantine_status, shut_down, community_risk, time):
        self.student_status = student_status
        self.teacher_status = teacher_status
        self.course_quarantine_status = course_quarantine_status
        self.shut_down = shut_down
        self.community_risk = community_risk
        self.time = time

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

        new_state = CampusState.get_observation()

        return new_state

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

    def get_reward(self, previous_state):
        reward = 0
        return reward
