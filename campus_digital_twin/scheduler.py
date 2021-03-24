from collections import defaultdict
import pandas as pd


class CourseRoomScheduler():
    def __init__(self, room_capacity, students_per_course, courses_with_conflict):
        self.room_capacity = room_capacity
        self.students_per_course = students_per_course
        self.courses_with_conflict = courses_with_conflict

    def get_schedule(self, action):
        """
        action - list of classes with % of infected students
        returns: return: dict([key=room], value=[list_of_courses])
        """
        room_course_dict = defaultdict(list)
        """

        Note:
        don't schedule a class with zero students
        """
        allowed_students_per_course = {}
        for course, occupancy in enumerate(self.students_per_course):  # self.allowed_students_per_course
            allowed_students_per_course[course] = int(action[course] / 100 * self.students_per_course[course])
            for room, cap in enumerate(self.room_capacity):
                conflict_flag = False

                if ((occupancy / cap) * self.students_per_course[course]) < self.room_capacity[room]:
                    for c in room_course_dict[room]:
                        if self.courses_with_conflict[c][course] == True:
                            conflict_flag = True
                    if conflict_flag == False:
                        room_course_dict[room].append(course)
                        break

        schedule = []
        for room, courses in room_course_dict.items():
            for course in courses:
                students = allowed_students_per_course[course]
                schedule.append((room, students, course))
        schedule_df = pd.DataFrame(schedule, columns=['Room', 'Students', 'Course'])
        return schedule_df

# def schedule_class(room_capacity, students_per_course, courses_with_conflict):
#     """
#     return: dict([key=room], value=[list_of_courses])
#     """
#     room_class_dict = defaultdict(list)
#     for course, occupancy in enumerate(students_per_course):
#         for room, cap in enumerate(room_capacity):
#             conflict_flag = False
#             if ((occupancy / cap) * students_per_course[course]) < room_capacity[room]:
#                 for c in room_class_dict[room]:
#                     if courses_with_conflict[c][course] == True:
#                         conflict_flag = True
#                 if conflict_flag == False:
#                     room_class_dict[room].append(course)
#                     break
#
#     return room_class_dict

