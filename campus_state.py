from collections import defaultdict
import itertools as it
import campus_model as cm
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
        """
        :param num_week:
        :return: list of tuples (course, classroom_id)

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
        print (room_capacity)
        print (students_per_course)
        # print (courses_with_conflict)
        print ("--------------------")

        """
        Check if at the current occupancy level the class c can be scheduled in room r 
        # or not (i.e. does the room have enough space for that many students). 
        """
        room_class_dict = defaultdict(list)
        for room, cap in enumerate(room_capacity):
            if not room_class_dict:
                room_class_dict[room] = []
            for course, occupancy in enumerate(students_per_course):
                if((occupancy/cap) * students_per_course[course]) < room_capacity[room]:
                    # print("Course:", course, "cannot be scheduled in room:", room)
                    room_class_dict[room].append(course)
                else:
                    pass
        pair_courses_to_schedule = []
        for room, courses in room_class_dict.items():
            courses_pairs = [p for p in it.product(courses, repeat=2)]
            for pair in courses_pairs:
                if(pair[0] == pair[1]):
                    pass
                else:
                    pair_courses_to_schedule.append(pair)

        for time, conflict_matrix in courses_with_conflict.items():
            for pair in pair_courses_to_schedule:
                # print(conflict_matrix)
                if (conflict_matrix[pair[0], pair[1]] == True):
                    print("courses", pair, "cannot be scheduled in room: ", room, "at time", time)
                else:
                    print("courses", pair, "can be scheduled in room: ", room, "at time", time)

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


