#!/bin/bash

# Define the list of alpha values
alphas=(0.01 0.1 0.25 0.35 0.45 0.5 0.6 0.8 1.0)

# Loop through each alpha value
for alpha in "${alphas[@]}"
do
    echo "Running training with alpha = $alpha"
    python main.py train --alpha $alpha
    # Optionally, you can redirect output to a file
    # python main.py train --alpha $alpha > "output_alpha_$alpha.txt"
done

echo "All training runs completed."
