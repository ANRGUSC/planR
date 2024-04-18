#!/bin/bash

# Define the list of alpha values
alphas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Loop through each alpha value
for alpha in "${alphas[@]}"
do
    echo "Running training with alpha = $alpha"
    python main.py train --alpha $alpha --agent_type "dqn_custom"
    # Optionally, you can redirect output to a file
    # python main.py train --alpha $alpha > "output_alpha_$alpha.txt"
done

echo "All training runs completed."
