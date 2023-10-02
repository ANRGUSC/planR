# SafeCampus-RL

A simulation system designed to study and analyze the dynamics of infection spread in a campus setting. It's structured within a reinforcement learning framework, utilizing tools for modeling, simulation, interaction, and management.

## Table of Contents
1. [Description](#description)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Components](#components)
6. [Contributing](#contributing)
7. [License](#license)

## Description

The project serves to simulate, study, and analyze infection spread scenarios in a campus environment.
It enables control over the number of students attending courses while considering different risk levels and provides rewards based on the number of allowed students. 
It integrates with Weights & Biases (wandb) for analytics and logging.

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

- `mode`: It can be either 'train', 'eval', or 'sweep'.
- `--alpha`: It is an optional argument representing the reward parameter alpha, the default is 0.1.
- `--agent_type`: It is an optional argument representing the type of agent to use, the default is 'qlearning'.

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
