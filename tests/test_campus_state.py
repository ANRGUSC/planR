from campus_simulator import campus_state as cs


def test_schedule_class():
    campus_state_instance = cs.CampusState()
    classrooms_with_classes = campus_state_instance.schedule_class()
    classrooms = len(classrooms_with_classes.keys())
    max_classroom = campus_state_instance.model.total_rooms()
    all_courses = []
    result_courses = set()
    for room, courses in classrooms_with_classes.items():
        all_courses.append(courses)
        # Value of each key is a list
        assert isinstance(courses, list) == True

    # number of keys in dictionary is less than or equal to the total number of rooms
    assert classrooms <= max_classroom

    # a dictionary is returned
    assert isinstance(classrooms_with_classes, dict) == True

    # if len(all_courses) > 1:
    #     result_courses = set(all_courses[0].intersection(*all_courses[1:]))
    #     print(result_courses)
    # else:
    #     result_courses = set(all_courses[0])
    #     print(result_courses)

def test_number_of_students_per_course():
    campus_state_instance = cs.CampusState()
    students_per_course = campus_state_instance.model.number_of_students_per_course()
    room_capacity = campus_state_instance.model.room_capacity()
    for students in students_per_course:
        for room in room_capacity:
            assert students <= room



