
# Campus-v0

This is a reinforcement learning-based simulation tool that could be applied to suggest to campus operators how many 
students from each course to allow on a campus classroom each week. The tool aims to strike a balance between the 
conflicting goals of keeping students from getting infected, on one hand, and allowing more students to come into 
campus to allow them to benefit from in-person classes, on the other. 
It incorporates the following:
<ol>
<li>A general school that includes students, teachers, courses and classrooms</li>
<li>COVID-19 transmission model that estimates the number of infected students in an indoor room</li>
</ol>
The campus environment is implemented as a custom
[Gym](https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html) environment.

# Environmment description



## Instructions
### Step 1 (Optional) Setting up the development environment.
If you are new to python development, then this step will help you get started with setting up your environment.
You will need to have setup on your computer:
- Code editor (E.g Pycharm Community edition or Visual Studio Code)
- Ubuntu (latest version is ok)
- Miniconda

Once installed then do the following on your terminal.
```
# Create a conda environment
$ conda create -n envname
$ conda activate envname

# Download and install necessary packages
$ git clone https://github.com/ANRGUSC/planR.git
$ cd planR
$ pip install -r requirements.txt

```

### Step 2: Clone the repository to your machine and install the dependancies.
```
$ git clone https://github.com/ANRGUSC/planR.git
$ cd planR
$ pip install -r requirements.txt
```
### Step 3: Execute training. T
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











