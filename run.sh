#!/bin/bash

# Define the list of alpha values
alphas=(0.01 0.02 0.1 0.25 0.35 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.6 0.7 0.8 0.9 1.0)

# Loop through each alpha value
for alpha in "${alphas[@]}"
do
    echo "Running training with alpha = $alpha"
    python main.py train --alpha $alpha
    # Optionally, you can redirect output to a file
    # python main.py train --alpha $alpha > "output_alpha_$alpha.txt"
done

echo "All training runs completed."
