import gym
import campus_gym
from campus_digital_twin import simulation_engine as se
env = gym.make('campus-v0')
print(se.create_campus_state().model.room_capacity())
print(se.create_campus_state().model.number_of_students_per_course())
print(se.create_campus_state().course_quarantine_status)
print(se.create_campus_state().community_risk)
print(se.create_campus_state().model.is_conflict())
print(se.create_campus_state().update_with_infection_model())
print(se.get_action())






