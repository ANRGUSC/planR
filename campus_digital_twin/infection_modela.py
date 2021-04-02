def get_infected_students(allowed_number_of_students_per_course, community_risk):
    infected_students = []

    #print("Infected students: ", number_of_students_per_course)
    for i in allowed_number_of_students_per_course:
        infected_students.append(int(0.5 * pow(i, community_risk)))
    return infected_students


