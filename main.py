import os
import subprocess


def subprocess_cmd(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)


def run_simulation():
    working_dir = os.getcwd()
    try:

        os.chdir("data_generator")
        subprocess_cmd('python3 generate_simulation_params.py; python3 generate_model_csv_files.py')
        print("Dataset generated")
    except:
        print("Error generating dataset files")

    # start Training

    print("Starting Training")
    os.chdir(working_dir)
    os.chdir("campus_gym/campus_gym/envs")
    subprocess_cmd('python3 run.py')
    print("Check training and testing output on envs/results folder")


if __name__ == '__main__':
    run_simulation()
