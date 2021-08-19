import subprocess


def subprocess_cmd(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)


def run_simulation():
    try:
        subprocess_cmd('cd data-generator; python3 generate_simulation_params.py; python3 generate_model_csv_files.py')
        print("Dataset generated")

    except:
        print("Error generating dataset files")

    # start Training

    print("Starting Training")
    subprocess_cmd('cd campus_gym/campus_gym/envs; python3 basic-q-learning.py')
    print("Check training and testing output on envs folder")



if __name__ == '__main__':
    run_simulation()
