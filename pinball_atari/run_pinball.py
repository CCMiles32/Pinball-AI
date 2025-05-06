import gymnasium as gym
from gymnasium.utils import play
import ale_py
import time
import torch
from collections import deque
import numpy as np
import cv2 # ball_tracker might use cv2 directly

# --- Import necessary components from your project ---
from config import (
    NUM_AGENT_ACTIONS,
    ACTION_MAP,
    NUM_FRAMES_STACKED,
    IMAGE_SIZE # Used in preprocess_frame
    # Add any other config variables if your imported functions need them directly
)
from dqn_model import DQN
from preprocessing import preprocess_frame
from ball_tracker import detect_ball # If your model uses ball position

# --- Constants ---
MODEL_PATH = "weights/weights_20250505_233349/pinball_dqn_step_8604876.pth"  # <--- REPLACE THIS WITH THE ACTUAL PATH TO YOUR WEIGHTS FILE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RENDER_DELAY = 0.05  # Seconds to pause between frames for watchability

# --- Helper function to stack frames (copied from train.py for self-containment) ---
def get_stacked_frames_tensor(state_buffer):
    """Converts a deque of frames (numpy arrays) into a stacked tensor."""
    # Ensure frames are float32 before stacking for torch tensor conversion
    float_frames = [frame.astype(np.float32) for frame in state_buffer]
    stacked_frames_np = np.stack(list(float_frames), axis=0) # Shape: (NUM_FRAMES_STACKED, H, W)
    state_tensor = torch.tensor(stacked_frames_np, dtype=torch.float32).unsqueeze(0) # Shape: (1, NUM_FRAMES_STACKED, H, W)
    return state_tensor

def watch_trained_agent():
    print(f"Attempting to load model from: {MODEL_PATH}")
    print(f"Using device: {DEVICE}")

    try:
        env = gym.make('ALE/VideoPinball-v5', render_mode='human')
        print("Environment created successfully!")

        # Get frame dimensions for ball normalization (if used)
        # Reset once to get an observation to infer dimensions
        temp_obs, _ = env.reset()
        frame_height = temp_obs.shape[0]
        frame_width = temp_obs.shape[1]
        print(f"Detected Frame Dimensions: {frame_width}x{frame_height}")


        # --- Model Initialization and Loading ---
        model = DQN(num_actions=NUM_AGENT_ACTIONS).to(DEVICE)
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        except FileNotFoundError:
            print(f"ERROR: Model weights file not found at {MODEL_PATH}")
            print("Please ensure the MODEL_PATH variable is set correctly.")
            env.close()
            return
        except Exception as e:
            print(f"Error loading model weights: {e}")
            env.close()
            return

        model.eval() # Set model to evaluation mode (disables dropout, etc.)
        print("Model loaded and set to evaluation mode.")

        # --- Game Loop ---
        for episode in range(10): # Run for a few episodes
            print(f"\nStarting Episode {episode + 1}")
            observation, info = env.reset()

            # --- Initialize State Processing ---
            processed_frame = preprocess_frame(observation)
            # The deque should store the preprocessed frames
            state_frame_buffer = deque([processed_frame] * NUM_FRAMES_STACKED, maxlen=NUM_FRAMES_STACKED)
            current_stacked_state_tensor = get_stacked_frames_tensor(state_frame_buffer)

            # Initialize ball state tensor (if your model uses it)
            ball_pos = detect_ball(observation) # observation is the raw frame from env
            if ball_pos:
                norm_ball_x = min(max(ball_pos[0] / frame_width, 0.0), 1.0)
                norm_ball_y = min(max(ball_pos[1] / frame_height, 0.0), 1.0)
            else:
                norm_ball_x = -1.0  # Indicate ball not detected/off-screen
                norm_ball_y = -1.0
            current_ball_state_tensor = torch.tensor([[norm_ball_x, norm_ball_y]], dtype=torch.float32)
            # ---

            terminated = False
            truncated = False
            total_reward_episode = 0
            step_count = 0

            while not terminated and not truncated:
                # --- Action Selection ---
                with torch.no_grad(): # No need to calculate gradients during inference
                    current_stacked_state_tensor_dev = current_stacked_state_tensor.to(DEVICE)
                    current_ball_state_tensor_dev = current_ball_state_tensor.to(DEVICE)

                    # Ensure your model's forward pass matches this call
                    q_values = model(current_stacked_state_tensor_dev, current_ball_state_tensor_dev)

                action_idx = torch.argmax(q_values).item()
                env_action = ACTION_MAP[action_idx]

                # --- Environment Step ---
                next_observation, reward, terminated, truncated, info = env.step(env_action)
                total_reward_episode += reward
                step_count += 1

                # --- Update State for Next Iteration ---
                next_processed_frame = preprocess_frame(next_observation)
                state_frame_buffer.append(next_processed_frame)
                current_stacked_state_tensor = get_stacked_frames_tensor(state_frame_buffer)

                # Update ball state tensor for the next iteration
                next_ball_pos = detect_ball(next_observation)
                if next_ball_pos:
                    next_norm_ball_x = min(max(next_ball_pos[0] / frame_width, 0.0), 1.0)
                    next_norm_ball_y = min(max(next_ball_pos[1] / frame_height, 0.0), 1.0)
                else:
                    next_norm_ball_x = -1.0
                    next_norm_ball_y = -1.0
                current_ball_state_tensor = torch.tensor([[next_norm_ball_x, next_norm_ball_y]], dtype=torch.float32)
                # ---

                observation = next_observation # For the next loop's ball detection if needed directly

                time.sleep(RENDER_DELAY)

                if terminated or truncated:
                    print(f"Episode finished after {step_count} steps. Total Reward: {total_reward_episode:.2f}")

            if not (terminated or truncated): # Safety break for very long episodes if needed
                print(f"Episode reached max steps without termination. Total Reward: {total_reward_episode:.2f}")


    except gym.error.Error as e:
        print(f"\nGymnasium Error: {e}")
        print("Ensure Atari ROMs are correctly installed/imported if this is a ROM issue.")
        print("Try: `python -m ale_py --import-roms /path/to/your/roms/directory`")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if 'env' in locals() and env is not None:
            env.close()
            print("Environment closed.")

if __name__ == "__main__":
    watch_trained_agent()