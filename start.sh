# Start Virtual Environment
source venv/bin/activate

# Generate the necessary campus simulator parameters
python3 generate_simulation_params.python3

# Start the simulator

python3 simulation_engine.py
