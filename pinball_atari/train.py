# pinball_ai/train.py

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import logging
import datetime
import os
from collections import deque # Import deque
import math

# --- Import necessary components ---
from config import * 
from dqn_model import DQN 
from utils import ReplayMemory
from preprocessing import preprocess_frame
from ball_tracker import detect_ball

# --- Constants ---
EPISODES_BETWEEN_DEEP_LOG = 100
EPISODE_SAVE_FREQUENCY = 100 # Frequency to save based on episodes
STEP_SAVE_FREQUENCY = 100000 # Frequency to save based on total steps
TREND_WINDOW_SIZE = 100
FLAT_THRESHOLD = 1e-5

# --- Configure Logging ---
log_dir = "logs"; os.makedirs(log_dir, exist_ok=True)
now = datetime.datetime.now(); timestamp_str = now.strftime("%Y%m%d_%H%M%S")
log_filename_only = f"training_log_{timestamp_str}.log"
log_file_full_path = os.path.join(log_dir, log_filename_only)
logging.basicConfig(filename=log_file_full_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
# Set INFO for file, but allow DEBUG level messages potentially for life loss detection
logger = logging.getLogger(__name__); logger.setLevel(logging.DEBUG) # Changed to DEBUG to allow life loss logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO); formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
# Ensure handlers are not added multiple times if script is re-run in interactive session
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(console_handler)
# Clear existing file handlers if any, to avoid writing to old files on re-runs
logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
file_handler = logging.FileHandler(log_file_full_path, mode='w')
file_handler.setLevel(logging.INFO) # Keep file log at INFO level
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

print(f"Logging to file: {log_file_full_path}") # Keep this print statement

# --- Helper function to stack frames from buffer ---
def get_stacked_frames_tensor(state_buffer):
    """Converts a deque of frames (numpy arrays) into a stacked tensor."""
    stacked_frames_np = np.stack(list(state_buffer), axis=0) # Shape: (NUM_FRAMES_STACKED, H, W)
    state_tensor = torch.tensor(stacked_frames_np, dtype=torch.float32).unsqueeze(0) # Shape: (1, NUM_FRAMES_STACKED, H, W)
    return state_tensor

# --- choose_action function (takes stacked_state_tensor) ---
def choose_action(model, stacked_state_tensor, state_ball, epsilon, device):
    if random.random() < epsilon:
        return random.randint(0, NUM_AGENT_ACTIONS - 1)
    else:
        with torch.no_grad():
            stacked_state_tensor = stacked_state_tensor.to(device)
            state_ball = state_ball.to(device)
            q_values = model(stacked_state_tensor, state_ball) # Model forward pass uses stacked frames
        return torch.argmax(q_values).item()

# --- optimize_model function (Receives stacked tensors via batch) ---
def optimize_model(model, target_model, optimizer, batch, device):
    stacked_states, state_balls, actions, rewards, next_stacked_states, next_state_balls, dones = zip(*batch)
    stacked_states = torch.cat(stacked_states).to(device)
    next_stacked_states = torch.cat(next_stacked_states).to(device)
    state_balls = torch.cat(state_balls).to(device)
    next_state_balls = torch.cat(next_state_balls).to(device)
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
    q_values = model(stacked_states, state_balls).gather(1, actions)
    with torch.no_grad():
        next_q_values_target = target_model(next_stacked_states, next_state_balls)
        best_next_q_values = next_q_values_target.max(1)[0].unsqueeze(1)
    expected_q_values = rewards + (GAMMA * best_next_q_values * (1.0 - dones))
    loss = F.smooth_l1_loss(q_values, expected_q_values)
    optimizer.zero_grad(); loss.backward()
    # --- Uncomment the line below to enable gradient clipping ---
    # torch.nn.utils.clip_grad_value_(model.parameters(), GRADIENT_CLIP_VALUE)
    optimizer.step()
    return loss.item()

# --- Helper function for saving model state ---
def save_model_state(model, save_path, step_or_episode_info):
    """Saves the model state_dict with logging and error handling."""
    try:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model state dictionary saved ({step_or_episode_info}) to {save_path}")
    except Exception as e:
        logger.error(f"Error saving model ({step_or_episode_info}) to {save_path}: {e}")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Using NUM_FRAMES_STACKED = {NUM_FRAMES_STACKED}")

    # --- Create Timestamped Weights Directory for this Run (inside base 'weights' folder) ---
    base_weights_dir = "weights"
    os.makedirs(base_weights_dir, exist_ok=True) # Ensure base 'weights' folder exists
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create path like weights/weights_YYYYMMDD_HHMMSS
    run_weights_dir = os.path.join(base_weights_dir, f"weights_{run_timestamp}")
    os.makedirs(run_weights_dir, exist_ok=True) # Create the specific run's weights folder
    logger.info(f"Saving weights for this run to: {run_weights_dir}")
    # ---

    # Ensure LIFE_LOST_PENALTY is accessible, provide default if not in config
    life_lost_penalty_value = LIFE_LOST_PENALTY
    logger.info(f"Using LIFE_LOST_PENALTY = {life_lost_penalty_value}")
    # Also log action costs being used
    logger.info(f"Action Costs: LEFT={LEFT_FLIPPER_COST}, RIGHT={RIGHT_FLIPPER_COST}, BOTH={BOTH_FLIPPERS_COST}, DOWN={DOWN_COST}, NO_FLIPPER={NO_FLIPPER_COST}, FIRE={FIRE_COST}")


    env = gym.make("ALE/VideoPinball-v5", render_mode="rgb_array")

    try:
        frame_height = env.observation_space.shape[0]; frame_width = env.observation_space.shape[1]
        logger.info(f"Detected Frame Dimensions: {frame_width}x{frame_height}")
    except Exception as e:
        logger.warning(f"Could not automatically get frame dimensions: {e}. Using defaults (e.g., 210x160)")
        frame_height = 210; frame_width = 160

    num_actions = NUM_AGENT_ACTIONS
    logger.info(f"Agent's Learned Action Space Size: {num_actions}")
    logger.info(f"Action Map (Agent Index -> Env Action): {ACTION_MAP}")

    model = DQN(num_actions).to(device)
    target_model = DQN(num_actions).to(device)
    target_model.load_state_dict(model.state_dict()); target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPSILON_START
    total_steps = 0
    last_step_save = 0 # Keep track of last step saved to avoid double saving if episode ends exactly on step boundary

    logger.info("Starting training...")
    for episode in range(1, NUM_EPISODES + 1):
        total_reward_episode = 0.0; current_loss = 0.0; num_optim_steps = 0
        done = False

        # --- Reset environment and get initial lives ---
        observation, info = env.reset()
        current_lives = info.get("lives", 0) # Get initial lives count
        #logger.debug(f"Episode {episode} Start: Initial Lives = {current_lives}")
 

        processed_frame = preprocess_frame(observation)
        state_frame_buffer = deque([processed_frame] * NUM_FRAMES_STACKED, maxlen=NUM_FRAMES_STACKED)
        current_stacked_state_tensor = get_stacked_frames_tensor(state_frame_buffer)

        ball_pos = detect_ball(observation)
        if ball_pos: norm_ball_x=min(max(ball_pos[0]/frame_width,0.0),1.0); norm_ball_y=min(max(ball_pos[1]/frame_height,0.0),1.0)
        else: norm_ball_x = -1.0; norm_ball_y = -1.0
        current_ball_state_tensor = torch.tensor([[norm_ball_x, norm_ball_y]], dtype=torch.float32)

        episode_steps = 0

        if episode % EPISODES_BETWEEN_DEEP_LOG == 0: logger.info(f"--- Episode {episode} Gameplay Start (Detailed Logging Enabled) ---")

        while not done:
            action_idx = choose_action(model, current_stacked_state_tensor, current_ball_state_tensor, epsilon, device)
            env_action = ACTION_MAP[action_idx]

            next_observation, reward_from_env, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated # Combined done condition

            # --- Check for Life Lost and Apply Penalty ---
            new_lives = info.get("lives", current_lives) # Get lives count after step
            life_lost_penalty = 0 # Default to no penalty
            if new_lives < current_lives:
                life_lost_penalty = life_lost_penalty_value # Apply penalty from config (or default)
                #logger.debug(f"--- LIFE LOST DETECTED (Step {total_steps})! Prev Lives: {current_lives}, New Lives: {new_lives}. Applying Penalty: {life_lost_penalty} ---")
            current_lives = new_lives # Update lives count for the next iteration
          

            # --- Calculate reward: Env Reward + Life Penalty + Action Cost ---
            current_reward = reward_from_env + life_lost_penalty # Start with env reward + life penalty

            action_name = AGENT_ACTION_SPACE[action_idx] # Get action name for cost lookup and logging
            action_cost = 0.0 # Default action cost
            if action_idx == 2: action_cost = LEFT_FLIPPER_COST # Index for LEFT_FLIPPER
            elif action_idx == 3: action_cost = RIGHT_FLIPPER_COST # Index for RIGHT_FLIPPER
            elif action_idx == 4: action_cost = BOTH_FLIPPERS_COST # Index for BOTH_FLIPPERS
            elif action_idx == 5: action_cost = DOWN_COST # Index for DOWN
            elif action_idx == 0: action_cost = NO_FLIPPER_COST # Index for NO_ACTION
            elif action_idx == 1: action_cost = FIRE_COST # Index for FIRE

            current_reward += action_cost # Add action cost


            # Deep logging moved here to show final calculated reward and penalty effect
            if episode % EPISODES_BETWEEN_DEEP_LOG == 0:
                 logger.info(f"   Step {episode_steps}: Epsilon={epsilon:.4f}, EnvReward={reward_from_env:.2f}, LifePenalty={life_lost_penalty}, ActionCost({action_name})={action_cost:.2f}, FinalReward={current_reward:.2f}, Lives: {current_lives}")


            next_processed_frame = preprocess_frame(next_observation)
            state_frame_buffer.append(next_processed_frame)
            next_stacked_state_tensor = get_stacked_frames_tensor(state_frame_buffer)

            next_ball_pos = detect_ball(next_observation)
            if next_ball_pos: next_norm_ball_x=min(max(next_ball_pos[0]/frame_width,0.0),1.0); next_norm_ball_y=min(max(next_ball_pos[1]/frame_height,0.0),1.0)
            else: next_norm_ball_x = -1.0; next_norm_ball_y = -1.0
            next_ball_state_tensor = torch.tensor([[next_norm_ball_x, next_norm_ball_y]], dtype=torch.float32)

            # Store the final 'current_reward' (including penalties/costs) in memory
            memory.push((current_stacked_state_tensor, current_ball_state_tensor, action_idx, current_reward,
                         next_stacked_state_tensor, next_ball_state_tensor, float(done)))

            current_stacked_state_tensor = next_stacked_state_tensor
            current_ball_state_tensor = next_ball_state_tensor
            observation = next_observation # Required for detect_ball in next iteration if needed directly

            # Log total reward based on the reward the agent optimizes (includes penalties/costs)
            total_reward_episode += current_reward
            total_steps += 1
            episode_steps += 1

            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                loss = optimize_model(model, target_model, optimizer, batch, device)
                current_loss += loss; num_optim_steps += 1

            if total_steps % TARGET_UPDATE_FREQ == 0:
                 logger.info(f"Updating target network at total step {total_steps}")
                 target_model.load_state_dict(model.state_dict())

            # # --- Step-based Model Saving ---
            # # Check if current step crossed a save frequency boundary
            # if total_steps // STEP_SAVE_FREQUENCY > last_step_save // STEP_SAVE_FREQUENCY:
            #      # Ensure we don't re-save if episode ends exactly on boundary and we also save by episode
            #      if total_steps != last_step_save:
            #          step_save_path = os.path.join(run_weights_dir, f"pinball_dqn_step_{total_steps}.pth")
            #          save_model_state(model, step_save_path, f"Step {total_steps}")
            #          last_step_save = total_steps # Update last saved step
            # # ---

            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # --- End of Episode ---
        avg_loss = current_loss / num_optim_steps if num_optim_steps > 0 else 0.0
        logger.info(f"Episode {episode}: Total Reward: {total_reward_episode:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.4f}, Steps: {episode_steps}, Total Steps: {total_steps}")

        # --- Episode-based Model Saving ---
        if episode % EPISODE_SAVE_FREQUENCY == 0:
            episode_save_path = os.path.join(run_weights_dir, f"pinball_dqn_episode_{episode}.pth")
            save_model_state(model, episode_save_path, f"Episode {episode}")
            # Update last_step_save to prevent immediate re-saving if step boundary also hit
            last_step_save = total_steps

        # --- Step-based Model Saving (Alternative Location: After Episode) ---
        # This ensures saving even if an episode ends exactly on the boundary,
        # complementing or replacing the episode-based saving.
        if total_steps // STEP_SAVE_FREQUENCY > last_step_save // STEP_SAVE_FREQUENCY:
             step_save_path = os.path.join(run_weights_dir, f"pinball_dqn_step_{total_steps}.pth")
             save_model_state(model, step_save_path, f"Step {total_steps}")
             last_step_save = total_steps # Update last saved step


    # --- End of Training ---
    env.close(); logger.info("Training finished.")

    # --- Save Final Model ---
    final_model_filename = "pinball_dqn_final.pth" # Simple name as it's inside timestamped folder
    final_save_path = os.path.join(run_weights_dir, final_model_filename)
    save_model_state(model, final_save_path, "Final")
    # ---

# if __name__ == "__main__":
#     # Ensure config values exist or set defaults (moved some from original spot)
#     globals().setdefault('NUM_EPISODES', 10000)
#     globals().setdefault('LEFT_FLIPPER_COST', -0.1) # Use values from your config if different
#     globals().setdefault('RIGHT_FLIPPER_COST', -0.1)
#     globals().setdefault('BOTH_FLIPPERS_COST', -0.2)
#     globals().setdefault('DOWN_COST', -0.5)
#     globals().setdefault('NO_FLIPPER_COST', 0.0)
#     globals().setdefault('FIRE_COST', -0.5)
#     globals().setdefault('LIFE_LOST_PENALTY', -50) # Added default for the new penalty
#     globals().setdefault('NUM_FRAMES_STACKED', 4)
#     globals().setdefault('GRADIENT_CLIP_VALUE', 1.0)
#     globals().setdefault('LEARNING_RATE', 1e-4)
#     globals().setdefault('GAMMA', 0.99)
#     globals().setdefault('EPSILON_START', 1.0)
#     globals().setdefault('EPSILON_MIN', 0.1)
#     globals().setdefault('EPSILON_DECAY', 0.999995)
#     globals().setdefault('BATCH_SIZE', 32)
#     globals().setdefault('MEMORY_SIZE', 100000)
#     globals().setdefault('TARGET_UPDATE_FREQ', 1000)
#     globals().setdefault('AGENT_ACTION_SPACE', ['NO_ACTION', 'FIRE', 'LEFT_FLIPPER', 'RIGHT_FLIPPER', 'BOTH_FLIPPERS', 'DOWN'])
#     globals().setdefault('NUM_AGENT_ACTIONS', len(AGENT_ACTION_SPACE))
#     globals().setdefault('ACTION_MAP', {0: 0, 1: 1, 2: 4, 3: 3, 4: 6, 5: 5})


#     # Ensure base log dir exists (weights dir is handled in train())
#     if not os.path.exists("logs"): os.makedirs("logs")

#     train()