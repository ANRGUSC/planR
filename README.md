# SafeCampus
A tool that simulates infection spread and facilitates the exploration of various RL algorithms in response to epidemic
challenges. The focus is in using reinforcement learning (RL) to develop occupancy
strategies that could balance minimizing infections with maximizing in-person inter-
actions in educational settings. This repository implements reinforcement learning agents. Checkout the 'Search' and 
'Myopic' sites to see alternative implementations that are non-RL.

[![Paper](https://img.shields.io/badge/Paper-Navigating_Safe_Campus_Operations_during_Epidemics-blue)](https://openreview.net/pdf?id=FudfN3ZJko)
## Table of Contents
3. [Installation](#installation)
4. [Usage](#usage)
5. [Components](#components)
6. [Contributing](#contributing)
7. [License](#license)


## Installation

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/ANRGUSC/SafeCampus-RL
   cd SafeCampus-RL
   ```
2. **Create a Conda Environment:**
   ```sh
   conda create --name myenv python=3.8
   conda activate myenv
   ```
3. **Install Required Packages:**
   ```sh
    pip install -r requirements.txt
    ```
## Components

### Main File
- `main.py`: This component coordinates the execution of the simulation including training, evaluation and sweeps.
### Model
- `model.py`: Contains the logic for estimating infected students based. This contains 2 different models.
    - `indoor-infection model`: Stochastic-based epidemic model based on the work by [this paper](https://www.pnas.org/doi/pdf/10.1073/pnas.2116165119). 
    - `Approximated SIR`: Based on the conventional SIR model and is used to estimate the number of infected students.
### Simulation
- `campus_state.py`: This performs the actual simulation of the campus scenario.
### Environment
- `campus_env.py`: This component defines the gymnasium environment for the simulation.
### Agent
- `agent.py`: The q_learning package is an example of how to implement an agent for this environment.

## Usage

To run the simulator, you can use the following command-line arguments:

- `mode`: It can be either 'train', 'eval', 'train' or 'sweep'.
- `--alpha`: It is an optional argument representing the reward parameter alpha, the default is 0.1.
- `--agent_type`: It is an optional argument representing the type of agent to use, the default is 'qlearning'.
- 
Default mode: Q learning 
Default environment: Discrete. Change this depending on agent and problem
   ```
      Update campus_gym_env.py in the campus_gym package.
      # For Deep RL
        # alpha = action[1]
        # self.campus_state.update_with_action(action[0])
        # observation = np.array(self.campus_state.get_student_status())

        # For Q-Learning
        alpha = action.pop()
        self.campus_state.update_with_action(action)
        observation = np.array(convert_actions_to_discrete(self.campus_state.get_student_status()))
    ```


Here are some examples of how to run the simulator:

1. **To run the training mode with default parameters:**
   ```sh
   python main.py train
    ```
2. **To run the sweep mode with a specific alpha and agent type:**
    ```sh
    python main.py sweep --alpha 0.2 --agent_type qlearning
     ```
## Visualization
After running the simulator, you can view the generated plots associated with a specific run_name 
to visualize the outcomes including the policy, Q-table, mean rewards with confidence intervals, and explained variance. 
Visualization files are located in:
```sh
<project-directory>/results/<run-name>.
```
1. **The outcome of the policy on all possible states, open the file located at:**
   ```sh
   <project-directory>/results/<run-name>/<max_episodes>-viz_all_states-<run_name>-<alpha>.png
    ```
2. ** Q-table, open the file located at:**
    ```sh
    <project-directory>/results/<run-name>/qtable-viz_q_table-<episode>.png
     ```
4. **Mean rewards with confidence intervals, open the file located at:**
    ```sh
    <project-directory>/results/<run-name>/<max_episodes>-viz_mean_rewards-<run_name>-<alpha>.png
     ```
5. **Explained variance, open the file located at:**
    ```sh
    <project-directory>/results/<run-name>/explained_variance-<episode>.png

     ```

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. 
We welcome contributions from anyone and everyone. Please refer to the [contributing guidelines](CONTRIBUTING.md) for more details.

## Future Work (TODO)
- [ ] Add more agent types.
- [ ] Add more simulation scenarios.
- [ ] Dockerization of the project is planned.
