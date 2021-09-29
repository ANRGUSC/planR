Status: 
# CampusPandemicPlanR
This is a tool for recommending allowable number of students taking different courses in a campus setting under pandemic uncertainties.
The campus twin environment is implemented as a [Gym](https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html) environment.
A paper explaining the details and results using a basic q_learning algorithm available at the [docs](docs/paper.pdf) folder.

## Running locally 
```
# Create a conda environment
$ conda create -n envname
$ conda activate envname

# Download and install necessary packages
$ git clone -b September https://github.com/ANRGUSC/planR.git
$ cd planR
$ pip install -r requirements.txt

# run
$ python3 main.py

```






