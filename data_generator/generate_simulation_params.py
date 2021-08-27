import yaml
import sys

students, teachers, classrooms, courses, weeks = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
params_data = [
    {'num_students': students},
    {'num_teacher': teachers},
    {'num_classes': classrooms},
    {'num_courses': courses},
    {'num_weeks': weeks}
]

with open('simulator_params.yaml', 'w') as file:
    document = yaml.dump(params_data, file)
