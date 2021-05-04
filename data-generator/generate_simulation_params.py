import yaml
params_data = [
    {'num_students': 100},
    {'num_teacher': 3},
    {'num_classes': 3},
    {'num_courses': 3},
    {'num_weeks': 15}
]

with open('simulator_params.yaml', 'w') as file:
    document = yaml.dump(params_data, file)
