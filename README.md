# Gettiing Started with CampusPandemicPlanR
This is a simulation environment for operating a campus under dynamic strategies to train reinforcement learning agents to autonomously recommend actions to administrators in an educational setting. It includes the following packages.

- **campus_digital_twin**: implements the general model of a school
- **campus_gym**: implements a custom Gym environment that includes an example using q-learning
- **data-generator**: used to generate simulation parameters and files necessary for the learning task.

## Features

- Generate simulation parameters
- Implement reinforcement learning algorithms for a general school model
- Adjust reward parameters for agent training.


## Installation and Running

CampusPandemicPlanR requires [Python 3+](https://www.python.org) to run.
Install the dependencies and devDependencies and start the server.

```sh
pip install -r requirements.txt 
cd data-generator
python3 gener
```
## Docker

CampusPandemicPlanR is very easy to install and deploy in a Docker container.

By default, the Docker will expose port 8080, so change this within the
Dockerfile if necessary. When ready, simply use the Dockerfile to
build the image.

## License

