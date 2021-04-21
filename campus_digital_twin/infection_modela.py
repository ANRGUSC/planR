def get_infected_students(current_infected_students,allowed_number_of_students_per_course, community_risk):
    # Hypothetical infection spread model {C(i)} = {C(i)^community_risk}.
    # Updated infection model: Infected students = c*(currently infected students*a + community_risk*allowed_students)*(allowed_students)
    # ~ c1*currently_infected_students*allowed_students + c2*community_risk*allowed_students^2.
    # For now, can pick c1 = 0.25 and c2 = 0.5.
    # c1 represents the probability that an infected students from the previous week shows up to class (i.e. is asymptomatic)
    # times the probability that they then go on to infect another student in class.
    # c2 represents the probability that a newly infected student infects another student in class.
    # Here currently_infected refers to the number of students infected the previous week
    infected_students = []
    c1 = 0.25
    c2 = 0.5

    for i in range(len(allowed_number_of_students_per_course)):
        infected = int((c1 * current_infected_students[i] * allowed_number_of_students_per_course[i]) + ( c2 * community_risk**i))
        percentage_infected = int(infected/allowed_number_of_students_per_course[i] * 100) if allowed_number_of_students_per_course[i] != 0 else 0
        infected_students.append(percentage_infected)
    # for i in allowed_number_of_students_per_course:
    #     for j in current_infected_students:
    #         infected_students.append(int((c1 * j * i) + (c2 * community_risk**i)))
    #         #infected_students.append(int(0.5 * pow(i, community_risk)))
    return infected_students


# infected_students = [0, 1, 4]
# allowed_students = [ 10, 20, 30]
# community_risk = 0.29
#
# print(get_infected_students(infected_students, allowed_students, community_risk))
