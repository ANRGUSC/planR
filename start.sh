# Start Virtual Environment
source venv/bin/activate

# Generate the necessary campus simulator parameters
python3 generate_model_csv_files.py

# Run Tests

python3 simulation_engine.py
