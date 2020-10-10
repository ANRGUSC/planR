import yaml
import load_data
import campus

# student_status = load_data.load_campus_data()[0]['infection']
# teacher_status = load_data.load_campus_data()[1]['']

with open(r'simulator_params.yaml') as file:
    sim_params_list = yaml.load(file, Loader=yaml.FullLoader)


class Simulator:

    def __init__(self, initialized=False, parameters=None):
        self.initialized = initialized
        self.parameters = parameters



