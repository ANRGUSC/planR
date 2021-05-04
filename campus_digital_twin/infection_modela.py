def get_infected_students(current_infected_students,allowed_number_of_students_per_course, community_risk):

    infected_students = []
    c1 = 0.25
    c2 = 0.5

    for i in range(len(allowed_number_of_students_per_course)):
        infected = int((c1 * current_infected_students[i] * allowed_number_of_students_per_course[i]) + ( c2 * community_risk**i))
        percentage_infected = int(infected/allowed_number_of_students_per_course[i] * 100) if allowed_number_of_students_per_course[i] != 0 else 0
        infected_students.append(percentage_infected)

    return infected_students
