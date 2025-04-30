import gymnasium as gym
import time # To add delays for visualization

print("Attempting to create the VideoPinball environment...")

try:
    # Create the Video Pinball environment
    # Use render_mode='human' to see the game window
    # Using the '-v5' version is generally recommended
    env = gym.make('VideoPinball-v5', render_mode='human')
    print("Environment created successfully!")

    # Reset the environment to get the initial state (observation)
    print("Resetting environment...")
    observation, info = env.reset()
    print("Environment reset.")
    print("Initial observation shape:", observation.shape) # Atari observations are typically images

    # Run the environment for a fixed number of steps (e.g., 1000)
    # In a real agent, this loop would run until a goal is met or learning converges
    for step in range(1000):
        # --- Action Selection (Replace with your AI/Agent logic later) ---
        # For now, just take a random action from the available actions
        action = env.action_space.sample()
        # print(f"Step: {step}, Action: {action}") # Uncomment to see actions

        # --- Environment Step ---
        # Apply the action and get the results
        # observation: The new state of the environment (screen pixels)
        # reward: The reward received for the action
        # terminated: Boolean, True if the episode ended naturally (e.g., game over)
        # truncated: Boolean, True if the episode ended due to a limit (e.g., time limit)
        # info: Dictionary with additional environment-specific information
        observation, reward, terminated, truncated, info = env.step(action)

        # Optional: Add a small delay so you can watch it
        time.sleep(0.05)

        # --- Episode End Check ---
        # If the episode is over (terminated or truncated), reset it
        if terminated or truncated:
            print(f"Episode finished after {step+1} steps. Resetting.")
            observation, info = env.reset()

    print("Finished simulation loop.")

except gym.error.Error as e:
    print(f"\nError creating or running the environment: {e}")
    print("\nTroubleshooting steps:")
    print("1. ROM Issue: Ensure you have the necessary Atari ROMs. While ale-py often downloads them, sometimes manual import is needed.")
    print("   You might need to find Atari 2600 ROMs (e.g., 'Video Pinball (1980) (Atari).a26') and import them using:")
    print("   `python -m ale_py --import-roms /path/to/your/roms/directory`")
    print("2. Environment ID: Double-check the environment ID ('VideoPinball-v5'). Sometimes older IDs like 'VideoPinballNoFrameskip-v4' might be needed, although v5 is preferred.")
    print("3. Installation: Verify gymnasium and ale-py are correctly installed in the 'pinball' environment.")

finally:
    # --- Cleanup ---
    # Always close the environment when you're done
    if 'env' in locals() and env is not None:
         env.close()
         print("Environment closed.")
    else:
         print("Environment was not successfully created, no need to close.")