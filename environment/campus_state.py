from collections import defaultdict
from environment import campus_model as cm


class CampusState():
    campus_state_data = {
        'student_status': [],
        'teacher_status': [],
        'course_quarantine': [],
        'shut_down': [],
        'community_risk': [],
        'course_operation_status': [],
        'classroom_schedule': []
    }
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
        }

        return observation

    def update_with_action(self, action):
        #TODO: (Elizabeth)
        self.course_operation_status = action

    def schedule_class(self):
        # TODO: This function needs to return the dict with list of courses
        """
        :param num_week:
        :return: current return is a dict with keys as rooms and values as courses fit for the room
        ---------------
        Note
        ---------------
        From meeting notes, the implementation was described as follows:
        1. Check if at the current occupancy level the class c can be scheduled in room r or not
        (i.e. does the room have enough space for that many students).
        2. Check if at the times that class c is happening, is there already a conflicting
        class scheduled in room c
        The checks seem to be done at
        """
        room_capacity = self.model.room_capacity()
        students_per_course = self.model.number_of_students_per_course()
        courses_with_conflict = self.model.is_conflict()
        room_course_list = []
        print(room_capacity)
        print(students_per_course)
        # print (courses_with_conflict)
        print("--------------------")

        """
        Check if at the current occupancy level the class c can be scheduled in room r 
        # or not (i.e. does the room have enough space for that many students). 
        """
        room_class_dict = defaultdict(list)
        for course, occupancy in enumerate(students_per_course):
            for room, cap in enumerate(room_capacity):
                # if not room_class_dict:
                #     room_class_dict[room] = []
                conflict_flag = False
                # room_class_dict[room] = []
                if ((occupancy / cap) * students_per_course[course]) < room_capacity[room]:
                    for c in room_class_dict[room]:

                        if courses_with_conflict[c][course] == True:
                            conflict_flag = True

                    if conflict_flag == False:
                        room_class_dict[room].append(course)
                        break

        return room_class_dict


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

x = CampusState()
print(x.schedule_class())


