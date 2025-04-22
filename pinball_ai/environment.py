import random
from config import ACTION_SPACE

def choose_action(model, state_image, state_ball, epsilon):
    if random.random() < epsilon:
        return random.randint(0, len(ACTION_SPACE)-1)
    else:
        q_values = model(state_image, state_ball)
        return torch.argmax(q_values).item()

def calculate_reward(events):
    reward = 0
    if events.get('ball_saved'):
        reward += 1
    if events.get('ball_drain'):
        reward -= 5
    return reward