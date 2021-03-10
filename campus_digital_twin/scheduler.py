from collections import defaultdict
class CourseRoomScheduler():
    def __init__(self, room_capacity, students_per_course, courses_with_conflict):
        self.room_capacity=room_capacity
        self.students_per_course = students_per_course
        self.courses_with_conflict = courses_with_conflict

    def get_schedule(self):
        """
        returns: return: dict([key=room], value=[list_of_courses])
        """
        room_course_dict = defaultdict(list)
        for course, occupancy in enumerate(self.students_per_course):
            for room, cap in enumerate(self.room_capacity):
                conflict_flag = False
                if ((occupancy / cap) * self.students_per_course[course]) < self.room_capacity[room]:
                    for c in room_course_dict[room]:
                        if self.courses_with_conflict[c][course] == True:
                            conflict_flag = True
                    if conflict_flag == False:
                        room_course_dict[room].append(course)
                        break
        return room_course_dict

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

