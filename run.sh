#!/bin/bash

# Define the list of alpha values
alphas=(0.01 0.02 0.1 0.25 0.35 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.6 0.7 0.8 0.9 1.0)

# Export main.py so it is available to subshells
export -f main.py

# Run tasks in parallel
printf "%s\n" "${alphas[@]}" | xargs -I {} -P 8 bash -c 'echo "Running training with alpha = {}"; python main.py train --alpha {}'

echo "All training runs completed."
