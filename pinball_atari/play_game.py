import gymnasium as gym
from gymnasium.utils import play
import ale_py

print('gym:', gym.__version__)
print('ale_py:', ale_py.__version__)

# Create Video Pinball environment
env = gym.make("ALE/VideoPinball-v5", render_mode="rgb_array")

print("Action space:", env.action_space)
print("Number of actions:", env.action_space.n)

# Basic key mapping (you can expand this with more actions)
keys_to_action = {
    (ord(' '),): 1,      # FIRE (usually launches the ball)
    (ord('a'),): 4,      # LEFT flipper
    (ord('d'),): 3,      # RIGHT flipper
    (ord('s'),): 5,      # DOWN (e.g. pull plunger)
    (ord('w'),): 2       # BOTH (may be used in nudging or menu)
}

# Start interactive play
play.play(env, keys_to_action=keys_to_action, zoom=3)
