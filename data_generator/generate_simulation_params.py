import random
import yaml


students = random.randint(100, 1000)
teachers = random.randint(3, 100)
classrooms = 3
courses = 3
weeks = 15
params_data = [
    {'num_students': students},
    {'num_teacher': teachers},
    {'num_classes': classrooms},
    {'num_courses': courses},
    {'num_weeks': weeks}
]

with open('simulator_params.yaml', 'w') as file:
    document = yaml.dump(params_data, file)
