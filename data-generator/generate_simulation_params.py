import yaml
import load_data


params_data = [
    {'num_students': 10},
    {'num_teacher': 3},
    {'num_classes': 3},
    {'num_courses': 3},
    {'num_weeks': 5},
    {'uninfected_known_students': 4},
    {'uninfected_unknown_students': 0},
    {'infected_known_students': 0},
    {'infected_unknown_students': 0},
    {'recovered_known_students': 0},
    {'recovered_unknown_students': 0},
    {'uninfected_known_teachers': 3},
    {'uninfected_unknown_teachers': 0},
    {'infected_known_teachers': 0},
    {'infected_unknown_teachers': 0},
    {'recovered_known_teachers': 0},
    {'recovered_unknown_teachers': 0},
    {'shutdown': [True, False, True, True, True]
     },
    {'community_risk': [0.3, 0.1, 9.2, 0.4, 0.5]},
    {'course_quarantine_status': [True, False, True, True, False, True]}  # The index represents course id
]

with open('simulator_params.yaml', 'w') as file:
    document = yaml.dump(params_data, file)
