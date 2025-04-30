# pinball_ai/train.py

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random # Import random for epsilon-greedy
import time


# --- Import necessary components ---
from config import * # Import all config variables
from dqn_model import DQN # Assumes DQN model is updated for image and ball state
from utils import ReplayMemory
from preprocessing import preprocess_frame
from ball_tracker import detect_ball # Import ball detection function

# --- Define choose_action logic here or import if kept separate ---
# It's often simpler to have it directly in the training loop if it depends
# closely on the model and state representation used here.
def choose_action(model, state_image, state_ball, epsilon, device):
    """
    Selects an action using epsilon-greedy strategy.
    Assumes model takes both image and ball state tensors.
    """
    if random.random() < epsilon:
        # Return a random action INDEX (0 to NUM_AGENT_ACTIONS-1)
        return random.randint(0, NUM_AGENT_ACTIONS - 1)
    else:
        with torch.no_grad():
            # Ensure tensors are on the correct device
            state_image = state_image.to(device)
            state_ball = state_ball.to(device)
            # Get Q-values from the model (which now takes both inputs)
            q_values = model(state_image, state_ball)
        # Return the index of the best action
        return torch.argmax(q_values).item()

# --- Updated optimize_model function ---
def optimize_model(model, target_model, optimizer, batch, device):
    """
    Optimizes the DQN model using a batch of experiences.
    Handles both image and ball state tensors.
    """
    # Unpack the batch correctly including ball states
    # Expected order: (state_img, state_ball, action_idx, reward, next_state_img, next_state_ball, done)
    state_images, state_balls, actions, rewards, next_state_images, next_state_balls, dones = zip(*batch)

    # Concatenate tensors correctly and send to device
    state_images = torch.cat(state_images).to(device)
    state_balls = torch.cat(state_balls).to(device) # Handle ball state
    next_state_images = torch.cat(next_state_images).to(device)
    next_state_balls = torch.cat(next_state_balls).to(device) # Handle next ball state

    # Convert actions, rewards, dones to tensors
    # Actions are indices (long), rewards/dones are floats
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1) # Use 0.0 and 1.0

    # --- Q-Value Calculation (using the updated model forward pass) ---
    q_values = model(state_images, state_balls).gather(1, actions)

    # --- Target Q-Value Calculation (using the updated target model forward pass) ---
    with torch.no_grad():
        # Get Q-values for the next states from the target network
        next_q_values_target = target_model(next_state_images, next_state_balls)
        # Select the best Q-value for the next state (Double DQN could use policy model here)
        best_next_q_values = next_q_values_target.max(1)[0].unsqueeze(1)

    # --- Expected Q-Value Calculation (Bellman Equation) ---
    # If done is 1.0, the future reward component is zero
    expected_q_values = rewards + (GAMMA * best_next_q_values * (1.0 - dones))

    # --- Loss Calculation ---
    # Using Smooth L1 Loss (Huber Loss) is common in DQN
    loss = F.smooth_l1_loss(q_values, expected_q_values)

    # --- Optimization ---
    optimizer.zero_grad()
    loss.backward()
    # Optional: Gradient clipping (uncomment if needed and GRADIENT_CLIP_VALUE is defined in config)
    # torch.nn.utils.clip_grad_value_(model.parameters(), GRADIENT_CLIP_VALUE)
    optimizer.step()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the correct environment ID and render mode
    # render_mode="human" for visualization, "rgb_array" for faster training
    env = gym.make("ALE/VideoPinball-v5", render_mode="rgb_array") 

    try:
        frame_height = env.observation_space.shape[0]
        frame_width = env.observation_space.shape[1]
        print(f"Detected Frame Dimensions: {frame_width}x{frame_height}")
    except Exception as e:
        print(f"Warning: Could not automatically get frame dimensions: {e}. Using defaults (e.g., 210x160)")
        # Fallback to default dimensions if detection fails
        frame_height = 210
        frame_width = 160

    # Agent's action space size
    num_actions = NUM_AGENT_ACTIONS
    print(f"Agent's Learned Action Space Size: {num_actions}")
    print(f"Action Map (Agent Index -> Env Action): {ACTION_MAP}")

    # Initialize models
    model = DQN(num_actions).to(device)
    target_model = DQN(num_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval() # Target model is only for inference

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPSILON_START
    total_steps = 0

    print("Starting training...")
    for episode in range(1, NUM_EPISODES + 1):
        total_reward = 0
        done = False
        observation, info = env.reset() # Get initial observation (raw frame)
        episode_steps = 0

        
        # --- Initial State Processing ---
        # 1. Detect ball in the raw frame
        ball_pos = detect_ball(observation)
        
        # 2. Normalize ball coordinates (0 to 1) and handle None
        if ball_pos:
            norm_ball_x = min(max(ball_pos[0] / frame_width, 0.0), 1.0) # Clamp to [0, 1]
            norm_ball_y = min(max(ball_pos[1] / frame_height, 0.0), 1.0)
        else:
            norm_ball_x = -1.0 # Use a distinct value if ball not found
            norm_ball_y = -1.0

        # Create ball state tensor (add batch dimension)
        state_ball_tensor = torch.tensor([[norm_ball_x, norm_ball_y]], dtype=torch.float32) # Keep on CPU for memory

        # 3. Preprocess the raw frame for the CNN
        state_image = preprocess_frame(observation) # Returns (H, W) numpy array
        # Add batch and channel dimensions for CNN: [1, 1, H, W]
        state_image_tensor = torch.tensor(state_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Keep on CPU for memory

        print(f"\n--- Episode {episode} Gameplay Start ---") # Mark start of agent control


        while not done:
            # --- Action Selection ---
            # Choose action index using epsilon-greedy based on current state tensors
            # Pass tensors to device within choose_action or before calling
            action_idx = choose_action(model, state_image_tensor, state_ball_tensor, epsilon, device)

            # Map agent action index to environment action ID
            env_action = ACTION_MAP[action_idx]
            next_observation, reward, terminated, truncated, info = env.step(env_action)
            raw_score = info.get("score", None)
            lives = info.get("lives", None)
            done = terminated or truncated # Episode ends if terminated or truncated
            #print(f"  Step {episode_steps}: Epsilon={epsilon:.4f}, Reward={reward:.2f}, Chosen Agent Action Index={AGENT_ACTION_SPACE[action_idx]}, Mapped Env Action={env_action}, ")
            #print(f"  Step {episode_steps}: Epsilon={epsilon:.4f}, Reward={reward:.2f}, Total Reward: {total_reward:.2f}, Chosen Agent Action Index={AGENT_ACTION_SPACE[action_idx]}, Mapped Env Action={env_action}")
            print(f"   Step {episode_steps}: Epsilon={epsilon:.4f}, Reward={reward:.2f}, Total Reward: {total_reward:.2f}, Lives: {lives}, Chosen Agent Action Index={AGENT_ACTION_SPACE[action_idx]}")

            # --- Environment Step ---
            if raw_score is not None:
                print(f"Game Score at Step {episode_steps}: {raw_score}")
            # --- Apply Action Cost (Optional) ---
            if action_idx == 2: # Index for 'LEFT_FLIPPER'
                FLIPPER_ACTION_COST = LEFT_FLIPPER_COST 
            elif action_idx == 3: # Index for 'RIGHT_FLIPPER'
                FLIPPER_ACTION_COST = RIGHT_FLIPPER_COST 
            elif action_idx == 4: # Index for 'BOTH_FLIPPERS'
                FLIPPER_ACTION_COST = BOTH_FLIPPERS_COST 
            elif action_idx == 5: # Index for 'DOWN'
                FLIPPER_ACTION_COST = DOWN_COST 
            elif action_idx == 0: # Index for 'NO_ACTION'
                FLIPPER_ACTION_COST = NO_FLIPPER_COST 
            elif action_idx == 1: # Index for 'FIRE'
                FLIPPER_ACTION_COST = NO_FLIPPER_COST 
            
            
            reward += FLIPPER_ACTION_COST # Add penalty

            # --- Process Next State ---
            # 1. Detect ball in the *next* raw frame
            next_ball_pos = detect_ball(next_observation)

            # 2. Normalize next ball coordinates
            if next_ball_pos:
                next_norm_ball_x = min(max(next_ball_pos[0] / frame_width, 0.0), 1.0)
                next_norm_ball_y = min(max(next_ball_pos[1] / frame_height, 0.0), 1.0)
            else:
                next_norm_ball_x = -1.0 # Use distinct value
                next_norm_ball_y = -1.0

            # Create next ball state tensor
            next_state_ball_tensor = torch.tensor([[next_norm_ball_x, next_norm_ball_y]], dtype=torch.float32) 

            # 3. Preprocess the *next* raw frame
            next_state_image = preprocess_frame(next_observation)
            # Add batch and channel dimensions
            next_state_image_tensor = torch.tensor(next_state_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 

            # --- Store Transition in Replay Memory ---
            memory.push((state_image_tensor, state_ball_tensor, # Current state
                         action_idx, reward,                   # Action index and reward
                         next_state_image_tensor, next_state_ball_tensor, # Next state
                         float(done)))                         # Done flag as float (0.0 or 1.0)

            # --- Update State for Next Iteration ---
            state_image_tensor = next_state_image_tensor
            state_ball_tensor = next_state_ball_tensor
            observation = next_observation # Keep raw frame if needed for next ball detection

            total_reward += reward
            total_steps += 1
            episode_steps += 1

            # --- Optimize Model ---
            # Start optimizing only when memory has enough samples
            if len(memory) > BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                optimize_model(model, target_model, optimizer, batch, device)

            # --- Update Target Network ---
            # Periodically copy weights from policy network to target network
            if total_steps % TARGET_UPDATE_FREQ == 0:
                 print(f"Updating target network at step {total_steps}")
                 target_model.load_state_dict(model.state_dict())

            # --- Epsilon Decay ---
            # Decay epsilon linearly or exponentially
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY) # Exponential decay example

            # Optional: Render environment for debugging/visualization
            # env.render() # Uncomment if using render_mode='human' or need explicit render

        # --- End of Episode ---
        print(f"Episode {episode}: Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}, Steps: {episode_steps}, Total Steps: {total_steps}")

        # --- Model Saving ---
        SAVE_FREQUENCY = 100 #Save every 100 episodes
        if episode % SAVE_FREQUENCY == 0:
            save_path = f"pinball_dqn_episode_{episode}.pth"
            try:
                 torch.save(model.state_dict(), save_path)
                 print(f"Model state dictionary saved to {save_path}")
            except Exception as e:
                 print(f"Error saving model at episode {episode}: {e}")

    # --- End of Training ---
    env.close()
    print("Training finished.")

    # --- Save Final Model ---
    final_save_path = "pinball_dqn_final.pth"
    try:
        torch.save(model.state_dict(), final_save_path)
        print(f"Final model state dictionary saved to {final_save_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")


# main to call train if this file is run directly
if __name__ == "__main__":
    if 'NUM_EPISODES' not in globals():
        print("Warning: NUM_EPISODES not found in config, using default 1000")
        NUM_EPISODES = 1000 # Default if not in config
    train()