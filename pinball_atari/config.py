# pinball_ai/config.py

# Hyperparameters and settings

IMAGE_SIZE = (84, 84) # <-- Keep the updated size
NUM_FRAMES_STACKED = 4 # <-- Keep for frame stacking

# --- Action Space Definition ---
AGENT_ACTION_SPACE = ['NO_ACTION', 'FIRE', 'LEFT_FLIPPER', 'RIGHT_FLIPPER', 'BOTH_FLIPPERS', 'DOWN']
NUM_AGENT_ACTIONS = len(AGENT_ACTION_SPACE)
ACTION_MAP = {
    0: 0,   # Agent action 0 ('NO_ACTION') maps to env action 9 (NOFIRE)
    1: 1,   # Agent action 1 ('FIRE') maps to env action 1 (FIRE)
    2: 4,   # Agent action 0 ('LEFT_FLIPPER') maps to env action 8 (LEFTFIRE)
    3: 3,   # Agent action 1 ('RIGHT_FLIPPER') maps to env action 7 (RIGHTFIRE)
    4: 6,   # Agent action 2 ('BOTH_FLIPPERS') maps to env action 6 (UPFIRE)
    5: 5    # Agent action 5 ('DOWN') maps to env action 5 (DOWN)

}

#FLIPPER_ACTION_COST = -2
LEFT_FLIPPER_COST = -0.2
RIGHT_FLIPPER_COST = -0.2
BOTH_FLIPPERS_COST = -0.3
NO_FLIPPER_COST = -0.1
DOWN_COST = -0.5
FIRE_COST = -0.5
LIFE_LOST_PENALTY = -1000
# --- End Action Space Definition ---

LEARNING_RATE = 0.00005
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
# Might want to slow this down further, e.g., 0.9999 or 0.99995 was at .999
EPSILON_DECAY = 0.999998
BATCH_SIZE = 32       
MEMORY_SIZE = 100000  
TARGET_UPDATE_FREQ = 50000
GRADIENT_CLIP_VALUE = 1.0

NUM_EPISODES = 10000  
# --- End Add ---