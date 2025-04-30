# pinball_ai/environment.py
import random
import torch
from config import NUM_AGENT_ACTIONS 

# choose_action now returns an index from 0 to NUM_AGENT_ACTIONS-1
def choose_action(model, state, epsilon):
    if random.random() < epsilon:
        # Choose a random action from the *agent's* reduced action space
        return random.randint(0, NUM_AGENT_ACTIONS - 1)
    else:
        with torch.no_grad():
            q_values = model(state) # Model outputs Q-values for agent actions
        # Get the index of the best action in the agent's action space
        return torch.argmax(q_values).item()

# calculate_reward function remains but is unused in the current train.py loop
def calculate_reward(events):
    # This is not used if we wrap the reward in train.py
    reward = 0
    if events.get('ball_saved'):
        reward += 1
    if events.get('ball_drain'):
        reward -= 1000
    return reward