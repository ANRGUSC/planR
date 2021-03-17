def get_infected_students(number_of_students_per_course, community_risk):
    infected_students = []
    print("Infected students: ", number_of_students_per_course)
    for i in number_of_students_per_course:
        infected_students.append(int(float(i) * community_risk))
    return infected_students


