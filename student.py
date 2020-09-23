import pandas as pd

class Student:
    """ A student file with the

    """

    def __init__(self, student_file):
        self.student_file = student_file
        file_path = pd.read_csv(filepath_or_buffer=self.student_file,
                                header=[0, 1, 2, 3, 4, 5])  # URL string

        self.student_info = file_path['student_info'].to_list()
        self.initial_infection = file_path['initial_infection'].to_list()
        self.c1 = file_path['c1'].to_list()
        self.c2 = file_path['c2'].to_list()
        self.c3 = file_path['c3'].to_list()
        self.c4 = file_path['c4'].to_list()



