from campus_digital_twin import simulation_engine as sim
sim = sim.initial_campus_state
print("Initial Observation:", sim.get_observation())
action = [10, 35, 50, 10, 50]
sim.update_with_action(action)
print("Updated:", sim.get_observation())
action = [20, 35, 30, 10, 50]
sim.update_with_action(action)
print("Updated:", sim.get_observation())






