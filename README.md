
# Campdrl

This is a reinforcement learning-based simulation tool that could be applied to suggest to campus operators how many 
students from each course to allow on a campus classroom each week. The tool aims to strike a balance between the 
conflicting goals of keeping students from getting infected, on one hand, and allowing more students to come into 
campus to allow them to benefit from in-person classes, on the other. 
It incorporates the following:
<ol>
<li>A general school campus model that includes students, teachers, courses and classrooms</li>
<li>A COVID-19 transmission model that estimates the number of infected students in an indoor classroom</li>
</ol>

# Environmment description
The reinforcement learning environment *Campus-v0* is implemented as a custom 
[Gym](https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html) environment.
Both the action and state space are represented as multi-discrete space where each discrete space has 3 levels.
For the state space, the length represents the number of courses (and each level is an approximation of the 
infected occupants) and community risk
## Setup Instructions
### Step 1 (Optional) Setting up the development environment.
If you are new to python development, then this step will help you get started with setting up your environment.
You will need to have setup on your computer:
- Code editor (E.g Pycharm Community edition or Visual Studio Code)
- Ubuntu (latest version is ok)
- Miniconda

Once installed then do the following on your terminal.
```
# Create and start the virtual environment to a project 
directory of your choice

$ python3 -m venv <name_of_virtualenv>
$ source <name_of_virtualenv>/bin/activate

# Clone the repository and install necessary packages
$ git clone https://github.com/ANRGUSC/planR.git
$ cd planR
$ pip install -r requirements.txt

```
### Step 2: Execute training.
There are 3 different reinforcement learning agents examples that have been implemented. 
The default training uses tabular q learning algorithm.
- Tabular Q-Learning
- Deep Q-learning
- Deep Q-learning with experience replay
```
$ python3 main.py
```

### Step 4: Analyze agent performance using rewards

Run the evaluation script to generate a plot on an agent's training performance.
```
$ python3 evaluate.py

```
The training results are stored in a json file that can be accessed in the results folder.
```
$ cd results

```











