import os
import sys
import subprocess


def subprocess_cmd(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)


def run(training_name):
    working_dir = os.getcwd()
    try:

        os.chdir("data_generator")
        subprocess_cmd('python3 generate_simulation_params.py 100 3 3 3 15')
        subprocess_cmd('python3 generate_model_csv_files.py')
        print("Dataset generated")
    except:
        print("Error generating dataset files")

    # start Training
    os.chdir(working_dir)
    os.chdir("campus_gym/campus_gym/envs")
    command = ["python3 run.py", training_name]
    subprocess_cmd(command)


if __name__ == '__main__':
    run_name = sys.argv[0]
    run(run_name)
