import os
import subprocess
import time


def subprocess_cmd(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)


def run():
    """Each run generates random numbers used to generate input files
       for crating a campus model.

      Note that, during training, the initial state is random for each agent created.
    """
    working_dir = os.getcwd()
    try:

        os.chdir("data_generator")
        subprocess_cmd('python3 generate_simulation_params.py')
        subprocess_cmd('python3 generate_model_csv_files.py')
        print("Dataset generated")

    except:
        print("Error generating dataset files")

    # start Training
    os.chdir(working_dir)
    os.chdir("campus_gym/campus_gym/envs")
    command = ["python3 run.py"]
    subprocess_cmd(command)


if __name__ == '__main__':
    start_time = time.time()
    run()
    time_taken = str('The training took {} seconds'.format(round(time.time() - start_time), 2))
    with open('log.txt', 'w+') as f:
        f.write(time_taken)
        f.close()


