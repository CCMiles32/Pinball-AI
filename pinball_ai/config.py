# Hyperparameters and settings

IMAGE_SIZE = (256, 256)
ACTION_SPACE = ['NO_FLIP', 'LEFT_FLIPPER', 'RIGHT_FLIPPER', 'BOTH_FLIPPERS']

LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 1000