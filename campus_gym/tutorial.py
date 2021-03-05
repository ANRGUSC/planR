class SchoolEnvironment():
    def __init__(self):
        self.steps_semester = 15

    def get_observation(self) -> List(int):
        return [10, 25, 40, 20]