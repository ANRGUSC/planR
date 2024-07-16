def estimate_infected_students(current_infected, allowed_per_course, community_risk):

    infected_students = []

    for i in range(len(allowed_per_course)):
        const_1 = 0.005  # reduce this to a smaller value
        const_2 = 0.01  # reduce this value to be very small 0.01, 0.02
        infected = int(((const_1 * current_infected[i]) * (allowed_per_course[i])) + (
                (const_2 * community_risk) * allowed_per_course[i] ** 2))

        infected = min(infected, allowed_per_course[i])

        infected_students.append(infected)

    infected_students = list(map(int, list(map(round, infected_students))))
    return infected_students

def estimate_infected_students_sir(current_infected, allowed_per_course, community_risk):

    infected_students = []
    total_students = 100
    # Iterate over each course
    for i in range(len(allowed_per_course)):
        # Constants for infection rates inside and outside the course
        const_1 = 0.001  # reduce this to a smaller value
        const_2 = 0.0099  # reduce this value to be very small 0.01, 0.02

        # Recovery rate for infected students
        recovery_rate = 0.1

        # Calculate the number of susceptible students in the course
        susceptible = max(0, allowed_per_course[i] - current_infected[i])

        # Calculate the number of susceptible students in the course
        # susceptible = max(0, total_students - current_infected[i])

        # Estimate new infections within the course
        new_infected_inside = int(
            (const_1 * current_infected[i]) * (susceptible / 100) * susceptible * allowed_per_course[i])

        # Estimate new infections from the community
        new_infected_outside = int((const_2 * community_risk * allowed_per_course[i]) * susceptible)

        # Estimate recovered students
        recovered = max(int(recovery_rate * current_infected[i]), 0)

        # Calculate the total number of new infections
        total_infected = new_infected_inside + new_infected_outside
        # Calculate the total number of infected students after accounting for recoveries

        infected = min(current_infected[i] + int(total_infected) - recovered, allowed_per_course[i])

        # Append the result to the list of infected students
        infected_students.append(infected)

    # Round and convert the infected students to integers
    infected_students = list(map(int, list(map(round, infected_students))))

    return infected_students
