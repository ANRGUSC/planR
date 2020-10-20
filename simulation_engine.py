import yaml
import campus_state


def load_sim_params(params_yaml):
    with open(params_yaml, 'r') as stream:
        try:
            sim_params = yaml.safe_load(stream)
            # print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    return sim_params


def search_sim_params(params_list_of_dict, search_string):
    # for each item on list
    # return: the value and index matching the search string.
    data_list = []
    for i in params_list_of_dict:
        [[key, value]] = i.items()
        my_list = key.split("_")
        if search_string in my_list:
            data_list.append(i)

    return data_list


def create_campus_state():
    sim_params = load_sim_params('simulator_params.yaml')
    student_status = search_sim_params(sim_params, 'students')
    teacher_status = search_sim_params(sim_params, 'teachers')
    course_quarantine_status = search_sim_params(sim_params, 'course')
    shut_down = search_sim_params(sim_params, 'shutdown')
    community_risk = search_sim_params(sim_params, 'community')
    time = search_sim_params(sim_params, 'weeks')

    campus_state_obj = campus_state.CampusState(True, student_status, teacher_status, course_quarantine_status,
                                                shut_down,
                                                community_risk, time)

    return campus_state_obj


CampusState = create_campus_state()
print(CampusState.time[0].get("num_weeks"))
