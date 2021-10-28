# CampusPandemicPlanR
This is a reinforcement learning-based simulation tool that could be applied to suggest to campus operators how many 
students from each course to allow on a campus classroom each week. The tool aims to strike a balance between the 
conflicting goals of keeping students from getting infected, on one hand, and allowing more students to come into 
campus to allow them to benefit from in-person classes, on the other. 
It incorporates the following:
<ol>
<li>A general school that includes students, teachers, courses and classrooms</li>
<li>COVID-19 transmission model that estimates the number of infected students in an indoor room/li>
</ol>
The campus environment is implemented as a custom
[Gym](https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html) environment.

# Environmment description



## Instructions
Step 1: Clone the repository to your machine and install the dependancies.
```
$ git clone https://github.com/ANRGUSC/planR.git
$ cd planR
$ pip install -r requirements.txt
```
Step 2: Execute training. The default training uses tabular q learning algorithm.
```
$ python3 main.py
```
There are 3 different reinforcement learning algorithm examples that are provided:
- Tabular Q-Learning
- Deep Q-learning
- Deep
```
$ python3 main.py
```








