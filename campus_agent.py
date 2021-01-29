"""
A campus agent is one that maintains
a social distance of 6.71 feet while positioning itself on
a time series path.

A time series path could consist of a table with timestamped position
data.
"""
import numpy as np
import random


def explore_strategy(actions):
    """
    :return: action index from the q_table
    """
    return random.choice(actions)


class QLearningAgent:
    def __init__(self, state_space_size, action_space_size):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.LEARNING_RATE = 0
        self.DISCOUNT_FACTOR = 0

    def select_action(self, current_state, action_list, epsilon):
        """
        :param epsilon:
        :param current_state: an integer value that corresponds to a row in the q-table
        :param action_list: a list of integers corresponding to actions that can be performed
        :return: an integer value that corresponds to a column in the current state row
        """
        if np.random.random() > epsilon:
            return self.exploit_strategy(current_state)
        else:
            return explore_strategy(action_list)

    def exploit_strategy(self, state):
        """
        :return: action index from the q_table
        """
        # the last time I did this I used amax instead of argmax
        return np.argmax(self.q_table[state])


    def update_internal_state(self, new_state, state_action_index, reward):
        """
        Update the Q-table here
        The internal state of the agent is the Q table which is updated using the equation below.
        :return: nothing
        """

        # Maximum possible Q value in next step (for new state)
        max_future_q = np.max(self.q_table[new_state])

        # Current Q value (for current state and performed action)
        current_q = self.q_table[state_action_index[0], state_action_index[1]]

        # And here's our equation for a new Q value for current state and action
        new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward[0] + self.DISCOUNT_FACTOR
                                                                             * max_future_q)
        # Update Q table with new Q value
        self.q_table[state_action_index[0], state_action_index[1]] = new_q


class RandomAgent:

    def __init__(self):
        pass


class MonteCarloAgent:

    def __init__(self):
        pass


class DeepQlearningAgent:
    def __init__(self):
        pass

