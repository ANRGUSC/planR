Status: 
# CampusPandemicPlanR
This is a tool for recommending allowable number of students taking different courses in a campus setting under pandemic uncertainties.
The campus twin environment is implemented as a [Gym](https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html) environment.
A paper explaining the details and results using an [Epsilon Greedy](docs/epsilon-greedy.pdf) algorithm available at the [docs](docs/paper.pdf) folder.

An agent observes the current state of the environment represented as the percentage of infected students in taking a 
given course in a fixed classroom size. The environment transitions to a new state and returns a reward.

## Running locally 
```
# Create a conda environment
$ conda create -n envname
$ conda activate envname

# Download and install necessary packages
$ git clone https://github.com/ANRGUSC/planR.git
$ cd planR
$ pip install -r requirements.txt

# run train
$ python3 main.py

# 

```







